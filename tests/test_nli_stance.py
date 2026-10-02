import itertools
import json
import math
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from quality_uncertainty_medrag import nli_stance as nli


TEXTS = dict(question_stem="What intervention is indicated?", candidate_option_text="Treatment A",
             evidence_title="A biomedical study", evidence_abstract="The intervention was studied.")
IDS = dict(question_id="medqa-us-dev-000004", candidate_option_id="A", evidence_doc_id="123")


def config(labels=("entailment", "neutral", "contradiction"), positions=512):
    return SimpleNamespace(num_labels=3, id2label=dict(enumerate(labels)),
                           label2id={label: i for i, label in enumerate(labels)}, max_position_embeddings=positions)


@pytest.mark.parametrize("labels", list(itertools.permutations(nli.SEMANTIC_LABELS)))
def test_mapping_uses_verified_semantics_not_fixed_indices(labels):
    mapping = nli.validate_label_mapping(config(labels))
    for i, label in enumerate(labels):
        assert mapping[str(i)] == {"nli_label": label, "stance": nli.SEMANTIC_LABELS[label]}
    scores = nli.softmax_logits([0, 0, 3])
    result = nli.score_result([0, 0, 3], scores, mapping)
    assert result["argmax_label"] == nli.SEMANTIC_LABELS[labels[2]]


def test_mapping_normalizes_case_and_json_indices():
    cfg = SimpleNamespace(num_labels=3, id2label={"0": " CONTRADICTION ", "1": "ENTAILMENT", "2": "Neutral"},
                          label2id={"contradiction": 0, "entailment": 1, "neutral": 2})
    assert nli.validate_label_mapping(cfg)["0"]["stance"] == "CONTRADICT"


@pytest.mark.parametrize("cfg", [
    SimpleNamespace(num_labels=2, id2label={0: "entailment", 1: "neutral"}),
    config(("LABEL_0", "LABEL_1", "LABEL_2")),
    config(("entailment", "neutral", "neutral")),
    SimpleNamespace(num_labels=3, id2label={False: "entailment", 1: "neutral", 2: "contradiction"}),
    SimpleNamespace(num_labels=3, id2label={"00": "entailment", "1": "neutral", "2": "contradiction"}),
    SimpleNamespace(num_labels=3, id2label={0: "entailment", 1: "neutral", 3: "contradiction"}),
    SimpleNamespace(num_labels=3, id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                    label2id={"entailment": 2, "neutral": 1, "contradiction": 0}),
])
def test_invalid_label_mapping_fails_explicitly(cfg):
    with pytest.raises(nli.NLIProbeError) as error:
        nli.validate_label_mapping(cfg)
    assert error.value.status == "LABEL_MAPPING_FAILURE"


def test_hypothesis_is_one_fixed_versioned_deterministic_template():
    kwargs = dict(question_stem="  question\nmore clues  ", candidate_option_text="  candidate  ")
    expected = "The proposed answer is: candidate\nClinical question: question\nmore clues"
    assert nli.build_hypothesis(**kwargs) == nli.build_hypothesis(**kwargs) == expected
    assert nli.HYPOTHESIS_VERSION == "question-single-option-hypothesis-v1"


def test_premise_contains_full_title_and_abstract_without_selection():
    assert nli.build_premise(evidence_title=" title ", evidence_abstract=" full\nabstract ") == "title\n\nfull\nabstract"
    assert nli.build_premise(evidence_title="title", evidence_abstract="") == "title"


@pytest.mark.parametrize("field", nli.INPUT_FIELDS)
@pytest.mark.parametrize("invalid", [None, 7, [], {}])
def test_required_input_fields_are_real_strings(field, invalid):
    inputs = {**TEXTS, field: invalid}
    with pytest.raises(ValueError):
        nli.build_pair(**inputs)


@pytest.mark.parametrize("forbidden", ["answer", "answer_idx", "gold_answer", "options", "other_options", "metadata", "candidate_correctness"])
def test_no_extra_gold_or_options_fields_can_enter_pair(forbidden):
    with pytest.raises(TypeError):
        nli.build_pair(**TEXTS, **{forbidden: "SECRET_GOLD_OR_OTHER_OPTIONS"})


@pytest.mark.parametrize("logits", [[10000, 9999, 9998], [-10000, -10001, -10002], [0, 0, 0], [2.5, -.7, .4]])
def test_stable_softmax_normalizes_finite_logits(logits):
    scores = nli.softmax_logits(logits)
    assert all(0 <= x <= 1 for x in scores)
    assert sum(scores) == pytest.approx(1, abs=1e-12)
    shifted = nli.softmax_logits([x + 1000 for x in logits])
    assert scores == pytest.approx(shifted, abs=1e-12)


@pytest.mark.parametrize("invalid", [[math.nan, 0, 0], [math.inf, 0, 0], [-math.inf, 0, 0], [True, 0, 0], [0, 0], ["0", 0, 0]])
def test_invalid_logits_are_rejected(invalid):
    with pytest.raises(nli.NLIProbeError):
        nli.softmax_logits(invalid)


@pytest.mark.parametrize("scores,entropy", [([1, 0, 0], 0), ([1 / 3] * 3, 1), ([.5, .5, 0], math.log(2) / math.log(3))])
def test_entropy(scores, entropy):
    assert nli.normalized_entropy(scores) == pytest.approx(entropy)


@pytest.mark.parametrize("scores", [[.1, .1, .1], [-.1, .5, .6], [math.nan, 0, 1], [True, 0, 0]])
def test_invalid_scores_fail(scores):
    with pytest.raises(nli.NLIProbeError):
        nli.normalized_entropy(scores)


def test_scores_must_match_logits_and_exact_tie_is_explicit():
    mapping = nli.validate_label_mapping(config())
    with pytest.raises(nli.NLIProbeError):
        nli.score_result([5, 0, 0], [1 / 3] * 3, mapping)
    tied = nli.score_result([0, 0, 0], [1 / 3] * 3, mapping)
    assert tied["argmax_label"] is None and tied["is_tied"] is True
    assert set(tied["argmax_labels"]) == set(nli.STANCE_ORDER)
    assert tied["scores_are_calibrated_probabilities"] is False


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Tokenizer:
    model_max_length = 512
    is_fast = True

    def __init__(self):
        self.calls = []

    def num_special_tokens_to_add(self, pair):
        assert pair is True
        return 3

    def __call__(self, premise, hypothesis, **kwargs):
        self.calls.append((premise, hypothesis, kwargs))
        length = len(premise.split()) + len(hypothesis.split()) + 3
        if kwargs["truncation"]:
            length = min(length, kwargs["max_length"])
        ids = list(range(length))
        return {"input_ids": Tensor([ids])} if kwargs.get("return_tensors") else {"input_ids": ids}


@pytest.mark.parametrize("length,limit,truncated", [(2, 64, False), (100, 64, True)])
def test_original_and_final_truncation_lengths_recorded(length, limit, truncated):
    tokenizer = Tokenizer()
    encoded, meta = nli.tokenize_pair(tokenizer, config(positions=limit), premise="word " * length, hypothesis="candidate")
    assert meta["original_token_length"] == length + 4
    assert meta["final_token_length"] == min(length + 4, limit)
    assert meta["truncation_occurred"] is truncated
    assert meta["effective_max_length"] == limit
    assert tokenizer.calls[0][2]["truncation"] is False
    assert tokenizer.calls[1][2]["truncation"] == "longest_first"
    assert encoded["input_ids"].shape[-1] <= limit


def test_huge_tokenizer_sentinel_does_not_override_card_or_model_limit():
    tokenizer = Tokenizer()
    tokenizer.model_max_length = 10 ** 30
    assert nli.sequence_limit(tokenizer, config())[0] == 256
    tokenizer.model_max_length = 128
    assert nli.sequence_limit(tokenizer, config())[0] == 128


@pytest.mark.parametrize("limit", [3, 257, True, 256.0])
def test_unsafe_sequence_cap_rejected(limit):
    with pytest.raises(ValueError):
        nli.sequence_limit(Tokenizer(), config(), limit)


def test_wrong_final_length_is_not_silently_recorded_as_valid():
    class BadTokenizer(Tokenizer):
        def __call__(self, premise, hypothesis, **kwargs):
            return {"input_ids": Tensor([[1] * 300])} if kwargs.get("return_tensors") else {"input_ids": [1] * 300}
    with pytest.raises(nli.NLIProbeError):
        nli.tokenize_pair(BadTokenizer(), config(), premise="text", hypothesis="candidate")


def fake_identity():
    return {"model_id": nli.MODEL_ID, "model_revision": nli.MODEL_REVISION,
            "resolved_model_revision": nli.MODEL_REVISION,
            "tokenizer_revision": nli.MODEL_REVISION, "resolved_tokenizer_revision": nli.MODEL_REVISION,
            "tokenizer_settings": {"max_length": 256, "truncation": "longest_first"}}


class Backend:
    identity = fake_identity()
    label_mapping = nli.validate_label_mapping(config())

    def __init__(self):
        self.calls = []

    def infer(self, **pair):
        self.calls.append(pair)
        return {**nli.score_result([2, 0, -1], nli.softmax_logits([2, 0, -1]), self.label_mapping),
                "truncation": {"original_token_length": 30, "final_token_length": 30, "truncation_occurred": False}}


def test_cache_key_contains_all_identity_versions_and_text_hashes():
    pair = nli.build_pair(**TEXTS)
    context = nli.build_cache_context(ids=IDS, inputs=TEXTS, pair=pair, backend_identity=fake_identity())
    first = nli.text_hash(nli.canonical(context))
    assert context["hypothesis_version"] == nli.HYPOTHESIS_VERSION
    for field in nli.INPUT_FIELDS:
        assert context["input_sha256"][field] == nli.text_hash(TEXTS[field])
    for change in [dict(model_revision="a" * 40), dict(tokenizer_revision="b" * 40), dict(tokenizer_settings={"max_length": 128})]:
        changed = {**context, "backend_identity": {**fake_identity(), **change}}
        assert nli.text_hash(nli.canonical(changed)) != first
    assert nli.text_hash(nli.canonical({**context, "hypothesis_version": "future"})) != first
    changed_text = {**TEXTS, "candidate_option_text": "A different candidate"}
    changed = nli.build_cache_context(ids=IDS, inputs=changed_text, pair=nli.build_pair(**changed_text), backend_identity=fake_identity())
    assert nli.text_hash(nli.canonical(changed)) != first
    assert nli.text_hash(nli.canonical({**context, "evidence_doc_id": "456"})) != first


def test_exact_cached_output_reused_without_another_inference(tmp_path):
    backend = Backend()
    classifier = nli.NLIStanceClassifier(backend, cache_dir=tmp_path / "nli_v0")
    first = classifier.classify_texts(**IDS, **TEXTS)
    cache = tmp_path / "nli_v0" / (first["cache_key"] + ".json")
    original = cache.read_bytes()
    second = classifier.classify_texts(**IDS, **TEXTS)
    assert len(backend.calls) == 1
    assert backend.calls[0] == nli.build_pair(**TEXTS)
    assert second["cache_status"] == "HIT" and second["model_calls"] == 0
    assert first["nli_label_scores"] == second["nli_label_scores"]
    assert cache.read_bytes() == original
    assert first["pair"]["hypothesis"] == nli.build_hypothesis(question_stem=TEXTS["question_stem"], candidate_option_text=TEXTS["candidate_option_text"])


def test_corrupt_cache_is_preserved_and_not_regenerated(tmp_path):
    backend = Backend()
    classifier = nli.NLIStanceClassifier(backend, cache_dir=tmp_path)
    result = classifier.classify_texts(**IDS, **TEXTS)
    cache = tmp_path / (result["cache_key"] + ".json")
    data = json.loads(cache.read_text())
    data["result"]["raw_logits"][0] = 20
    cache.write_text(json.dumps(data))
    corrupted = cache.read_bytes()
    with pytest.raises(nli.NLIProbeError):
        classifier.classify_texts(**IDS, **TEXTS)
    assert len(backend.calls) == 1 and cache.read_bytes() == corrupted


@pytest.mark.parametrize("name", ["v1", "logprob_v0"])
def test_qwen_caches_are_never_used_or_modified(tmp_path, name):
    with pytest.raises(ValueError):
        nli.NLIStanceClassifier(Backend(), cache_dir=tmp_path / name)


def test_classifier_rejects_gold_and_other_options_before_backend_call(tmp_path):
    backend = Backend()
    classifier = nli.NLIStanceClassifier(backend, cache_dir=tmp_path)
    for forbidden in ("answer", "options", "metadata", "answer_idx", "candidate_correctness"):
        with pytest.raises(TypeError):
            classifier.classify_texts(**IDS, **TEXTS, **{forbidden: "secret"})
    assert backend.calls == []


@pytest.mark.parametrize("kwargs", [dict(model_revision="main"), dict(tokenizer_revision="main"),
                                     dict(model_revision="a" * 40), dict(model_id="a/different-model")])
def test_revisions_are_full_pinned_matching_hashes(kwargs):
    with pytest.raises(ValueError):
        nli.NLIModelSpec(**kwargs)


class Model:
    dtype = "torch.float32"

    def __init__(self):
        self.config = config()
        self.calls = []
        self.training = True

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, **encoded):
        self.calls.append(encoded)
        assert self.training is False
        return SimpleNamespace(logits=Tensor([[1.5, -2, .4]]))

    def parameters(self):
        return [SimpleNamespace(numel=lambda: 150)]


def fake_torch():
    return SimpleNamespace(manual_seed=lambda value: None, use_deterministic_algorithms=lambda value: None,
                           inference_mode=nullcontext, float64="float64", __version__="mock",
                           softmax=lambda tensor, dim: Tensor([nli.softmax_logits(tensor.values[0])]),
                           cuda=SimpleNamespace(is_available=lambda: False))


def test_transformers_backend_evaluates_logits_softmax_and_records_lengths():
    tokenizer, model = Tokenizer(), Model()
    backend = nli.TransformersNLIBackend(tokenizer=tokenizer, model=model, torch_module=fake_torch(),
        spec=nli.NLIModelSpec(), snapshot_identity=fake_identity())
    result = backend.infer(**nli.build_pair(**TEXTS))
    assert result["raw_logits"] == [1.5, -2, .4]
    assert sum(result["nli_label_scores"].values()) == pytest.approx(1)
    assert len(model.calls) == 1
    assert set(model.calls[0]) == {"input_ids"}
    assert result["truncation"]["truncation_occurred"] is False
    assert backend.identity["resolved_model_revision"] == nli.MODEL_REVISION


def test_mismatched_downloaded_revision_fails_before_inference():
    with pytest.raises(nli.NLIProbeError):
        nli.TransformersNLIBackend(tokenizer=Tokenizer(), model=Model(), torch_module=fake_torch(),
            spec=nli.NLIModelSpec(), snapshot_identity={**fake_identity(), "resolved_model_revision": "a" * 40})


@pytest.mark.parametrize("use_existing_hf_login", [False, True])
def test_real_loader_uses_pinned_safe_snapshot_and_required_auto_classes(tmp_path, monkeypatch, use_existing_hf_login):
    snapshot = tmp_path / "snapshots" / nli.MODEL_REVISION
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).write_bytes(b"test-only mocked snapshot")
    download_calls, auto_calls = [], []
    def download(**kwargs):
        download_calls.append(kwargs)
        return str(snapshot)
    def tokenizer_from(path, **kwargs):
        auto_calls.append(("tokenizer", path, kwargs))
        return Tokenizer()
    def model_from(path, **kwargs):
        auto_calls.append(("model", path, kwargs))
        return Model()
    transformers = SimpleNamespace(__version__="mock", AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from),
                                   AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=model_from))
    class GatedRepoError(Exception):
        pass
    hub = SimpleNamespace(__version__="mock", snapshot_download=download)
    for name, module in {"torch": fake_torch(), "transformers": transformers, "huggingface_hub": hub,
                         "huggingface_hub.errors": SimpleNamespace(GatedRepoError=GatedRepoError),
                         "tokenizers": SimpleNamespace(__version__="mock"), "safetensors": SimpleNamespace(__version__="mock")}.items():
        monkeypatch.setitem(sys.modules, name, module)
    backend = nli.load_transformers_backend(hf_cache_dir=tmp_path / "cache", use_existing_hf_login=use_existing_hf_login)
    assert download_calls[0]["repo_id"] == nli.MODEL_ID
    assert download_calls[0]["revision"] == nli.MODEL_REVISION
    assert download_calls[0]["token"] is use_existing_hf_login
    assert backend.identity["hub_authentication_mode"] == ("existing_login" if use_existing_hf_login else "anonymous")
    assert auto_calls[0][2] == {"local_files_only": True, "use_fast": True, "trust_remote_code": False}
    assert auto_calls[1][2] == {"local_files_only": True, "use_safetensors": True, "trust_remote_code": False}
    assert backend.identity["resolved_tokenizer_revision"] == nli.MODEL_REVISION
    assert backend.identity["raw_id2label"] == {"0": "entailment", "1": "neutral", "2": "contradiction"}
    assert backend.identity["parameter_count"] == 150
    def denied(**kwargs):
        raise GatedRepoError("sensitive access token must never appear")
    hub.snapshot_download = denied
    with pytest.raises(nli.NLIProbeError) as error:
        nli.load_transformers_backend(hf_cache_dir=tmp_path / "cache", use_existing_hf_login=use_existing_hf_login)
    assert error.value.status == "MODEL_ACCESS_REQUIRED"
    assert "sensitive" not in str(error.value)

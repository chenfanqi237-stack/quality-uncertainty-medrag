import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from quality_uncertainty_medrag import nli_stance as nli
from quality_uncertainty_medrag import nli_truncation_control as control
from quality_uncertainty_medrag import stance_nli_probe as original


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
    model_max_length = 8192
    truncation_side = padding_side = "right"
    def __init__(self):
        self.calls = []
        self.backend_tokenizer = SimpleNamespace(
            truncation={"max_length": 256, "strategy": "LongestFirst"},
            padding={"strategy": {"Fixed": 256}})
    def num_special_tokens_to_add(self, pair):
        return 3
    def __call__(self, premise, hypothesis, **kwargs):
        self.calls.append((premise, hypothesis, kwargs))
        length = len(premise.split()) + len(hypothesis.split()) + 3
        if kwargs["truncation"]:
            length = min(length, kwargs["max_length"])
        ids = list(range(length))
        return {"input_ids": Tensor([ids])} if kwargs.get("return_tensors") else {"input_ids": ids}


def config(limit=8192):
    return SimpleNamespace(max_position_embeddings=limit, num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2})


class Model:
    def __init__(self):
        self.config = config()
        self.calls = []
    def to(self, device):
        return self
    def eval(self):
        self.training = False
    def __call__(self, **encoded):
        assert self.training is False
        self.calls.append(encoded)
        return SimpleNamespace(logits=Tensor([[1.5, -2.0, .4]]))


def base_backend():
    torch = SimpleNamespace(manual_seed=lambda seed: None, use_deterministic_algorithms=lambda flag: None,
        inference_mode=nullcontext, float64="float64",
        softmax=lambda logits, dim: Tensor([nli.softmax_logits(logits.values[0])]))
    identity = {"resolved_model_revision": nli.MODEL_REVISION, "resolved_tokenizer_revision": nli.MODEL_REVISION,
        "snapshot_file_sha256": {"model.safetensors": "mock-checksum"}, "runtime_versions": {"torch": "mock"},
        "model_dtype": "torch.float32", "logit_softmax_dtype": "torch.float64", "device": "cpu"}
    return nli.TransformersNLIBackend(tokenizer=Tokenizer(), model=Model(), torch_module=torch,
                                    spec=nli.NLIModelSpec(), snapshot_identity=identity)


@pytest.mark.parametrize("token_limit,model_limit,expected", [
    (8192, 8192, 8192), (10**30, 8192, 8192), (None, 8192, 8192),
    (4096, 8192, 4096), (8192, 512, 512)])
def test_largest_supported_finite_limit_without_training_cap(token_limit, model_limit, expected):
    tokenizer = Tokenizer()
    tokenizer.model_max_length = token_limit
    metadata = control.inspect_supported_context(tokenizer, config(model_limit))
    assert metadata["effective_max_length"] == expected
    assert metadata["loaded_native_truncation_before_control"]["max_length"] == 256
    assert metadata["loaded_native_padding_before_control"]["strategy"] == {"Fixed": 256}
    assert nli.MODEL_CARD_SEQUENCE_CAP == 256


@pytest.mark.parametrize("model_limit", [None, 10**30, -1, True, 8192.0, "8192", 3])
def test_unknown_or_unsafe_model_limit_fails_closed(model_limit):
    with pytest.raises(nli.NLIProbeError):
        control.inspect_supported_context(Tokenizer(), config(model_limit))


@pytest.mark.parametrize("words,limit,expected_removed", [(500,8192,0), (9000,8192,812)])
def test_serialized_256_limit_and_padding_are_explicitly_overridden(words, limit, expected_removed):
    tokenizer = Tokenizer()
    context = control.inspect_supported_context(tokenizer, config(limit))
    encoded, metadata = control.tokenize_supported_pair(tokenizer, premise="word " * words,
                                                       hypothesis="candidate", context=context)
    assert metadata["original_token_length"] == words + 4
    assert metadata["final_token_length"] == min(words + 4, limit)
    assert metadata["tokens_removed"] == expected_removed
    assert metadata["truncation_occurred"] == bool(expected_removed)
    assert tokenizer.calls[0][2]["truncation"] is False
    assert tokenizer.calls[1][2]["max_length"] == limit
    assert all(call[2]["padding"] is False for call in tokenizer.calls)


def test_unexpected_persistent_256_truncation_is_rejected():
    class WrongTokenizer(Tokenizer):
        def __call__(self, *args, **kwargs):
            if kwargs.get("return_tensors"):
                return {"input_ids": Tensor([[1] * 256])}
            return {"input_ids": [1] * 500}
    tokenizer = WrongTokenizer()
    with pytest.raises(nli.NLIProbeError, match="Final length"):
        control.tokenize_supported_pair(tokenizer, premise="evidence", hypothesis="candidate",
                                       context=control.inspect_supported_context(tokenizer, config()))


def test_adapter_reuses_exact_checkpoint_and_identical_math_for_short_pair():
    base = base_backend()
    old_identity = nli.canonical(base.identity)
    backend = control.SupportedContextNLIBackend(base)
    pair = {"premise": "Biomedical evidence", "hypothesis": "A single candidate hypothesis"}
    before = base.infer(**pair)
    after = backend.infer(**pair)
    for field in ("raw_logits", "nli_label_scores", "softmax_scores_in_model_label_order",
                  "normalized_entropy", "argmax_label"):
        assert before[field] == after[field]
    assert backend.model is base.model and backend.tokenizer is base.tokenizer
    assert backend.identity["max_length"] == 8192
    assert backend.identity["probe_version"] == control.CONTROL_VERSION
    assert nli.canonical(base.identity) == old_identity


def test_effective_limit_and_namespace_change_cache_identity(tmp_path):
    base = base_backend()
    backend = control.SupportedContextNLIBackend(base)
    ids = dict(question_id="medqa-us-dev-000004", candidate_option_id="A", evidence_doc_id="123")
    inputs = dict(question_stem="Question stem", candidate_option_text="Only candidate",
                  evidence_title="Evidence title", evidence_abstract="Evidence abstract")
    pair = nli.build_pair(**inputs)
    old_key = nli.text_hash(nli.canonical(nli.build_cache_context(ids=ids, inputs=inputs, pair=pair, backend_identity=base.identity)))
    new_key = nli.text_hash(nli.canonical(nli.build_cache_context(ids=ids, inputs=inputs, pair=pair, backend_identity=backend.identity)))
    assert old_key != new_key
    classifier = nli.NLIStanceClassifier(backend, cache_dir=tmp_path / control.CACHE_NAMESPACE)
    first = classifier.classify_texts(**ids, **inputs)
    cached = classifier.classify_texts(**ids, **inputs)
    assert first["cache_key"] == new_key
    assert cached["cache_status"] == "HIT" and cached["model_calls"] == 0
    assert len(base.model.calls) == 1


@pytest.fixture
def previous_run(tmp_path, monkeypatch):
    root = tmp_path / "project"
    source = root / control.PREVIOUS_RUN_RELATIVE
    source.mkdir(parents=True)
    backend = base_backend()
    inputs, pairs, previous = [], [], []
    for qid, option, pmid in original.DIAGNOSTIC_KEYS:
        row = dict(question_id=qid, candidate_option_id=option, evidence_doc_id=pmid,
            question_stem="What is indicated? " + "clinical " * 40, candidate_option_text="Only candidate " + option,
            evidence_title="Biomedical title", evidence_abstract="Scientific finding " * 200)
        pair = nli.build_pair(**{name: row[name] for name in nli.INPUT_FIELDS})
        ids = {name: row[name] for name in nli.ID_FIELDS}
        hashes = {name: nli.text_hash(row[name]) for name in nli.INPUT_FIELDS}
        result = {"status": "SUCCESS", **backend.infer(**pair), **ids, "pair": pair,
            "model_metadata": backend.identity, "input_sha256": hashes,
            "model_calls": 1, "cache_status": "MISS", "hard_label_agreement": False}
        inputs.append(row)
        pairs.append({**ids, **pair})
        previous.append(result)
    original.save_jsonl(source / "inputs.jsonl", inputs)
    original.save_jsonl(source / "pairs.jsonl", pairs)
    original.save_jsonl(source / "results.jsonl", previous)
    original.save_json(source / "model_metadata.json", backend.identity)
    monkeypatch.setattr(original, "DIAGNOSTIC_INPUT_SHA256", hashlib.sha256((source / "inputs.jsonl").read_bytes()).hexdigest())
    backend.model.calls.clear()
    return root, source, inputs, previous, backend


def test_full_control_uses_same_pairs_and_preserves_old_outputs_cache(previous_run):
    root, source, inputs, previous, backend = previous_run
    old_cache = root / "data/cache/stance/nli_v0/original.json"
    old_cache.parent.mkdir(parents=True)
    old_cache.write_text("old-cache-byte-identical", encoding="utf-8")
    before = {p: p.read_bytes() for p in source.iterdir()}
    loader_calls = []
    def loader(**kwargs):
        loader_calls.append(kwargs)
        return backend
    report, directory = control.run_control(root=root, backend_loader=loader)
    assert report["status"] == "COMPLETE" and report["model_calls"] == 18
    assert loader_calls[0]["local_files_only"] is True
    assert len(backend.model.calls) == 18
    assert report["summary"]["before"]["truncated_input_count"] == 18
    assert report["summary"]["after"]["truncated_input_count"] == 0
    assert report["summary"]["effective_max_length"] == 8192
    assert all(result["previous_final_token_length"] == 256 for result in report["results"])
    assert all(result["truncation"]["tokens_removed"] == 0 for result in report["results"])
    assert (directory / "pairs.jsonl").read_bytes() == (source / "pairs.jsonl").read_bytes()
    assert (directory / "inputs.jsonl").read_bytes() == (source / "inputs.jsonl").read_bytes()
    assert {p: p.read_bytes() for p in source.iterdir()} == before
    assert old_cache.read_text() == "old-cache-byte-identical"
    calls_before = len(backend.model.calls)
    report2, directory2 = control.run_control(root=root, backend_loader=loader)
    assert directory2 != directory
    assert report2["model_calls"] == 0 and report2["summary"]["cache_hits"] == 18
    assert len(backend.model.calls) == calls_before


@pytest.mark.parametrize("field", ["answer", "answer_idx", "options", "metadata", "candidate_correctness"])
def test_gold_or_other_options_in_saved_inputs_rejected(previous_run, monkeypatch, field):
    root, source, inputs, _, backend = previous_run
    inputs[0][field] = "GOLD_OTHER_OPTIONS_SECRET"
    original.save_jsonl(source / "inputs.jsonl", inputs)
    monkeypatch.setattr(original, "DIAGNOSTIC_INPUT_SHA256", hashlib.sha256((source / "inputs.jsonl").read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="four allowed"):
        control.run_control(root=root, backend_loader=lambda **kwargs: backend)
    assert backend.model.calls == []


def test_same_texts_only_are_passed_to_model(previous_run):
    root, source, inputs, _, backend = previous_run
    control.run_control(root=root, backend_loader=lambda **kwargs: backend)
    actual_pairs = [(premise, hypothesis) for premise, hypothesis, kwargs in backend.tokenizer.calls
                    if kwargs.get("return_tensors") and kwargs["max_length"] == 8192]
    expected_pairs = [(nli.build_pair(**{name: row[name] for name in nli.INPUT_FIELDS})["premise"],
                       nli.build_pair(**{name: row[name] for name in nli.INPUT_FIELDS})["hypothesis"]) for row in inputs]
    assert actual_pairs == expected_pairs
    assert all(set(kwargs) == {"input_ids"} for kwargs in backend.model.calls)


def test_saved_hypothesis_edits_are_rejected(previous_run):
    root, source, _, _, backend = previous_run
    pairs = control._lines(source / "pairs.jsonl")
    pairs[0]["hypothesis"] += " manual edit"
    original.save_jsonl(source / "pairs.jsonl", pairs)
    with pytest.raises(ValueError, match="no edits"):
        control.load_previous_run(source)
    assert backend.model.calls == []


def test_existing_destination_is_never_overwritten(previous_run):
    root, source, _, _, backend = previous_run
    target = root / control.OUTPUT_RELATIVE / "existing"
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="overwritten"):
        control.run_control(root=root, output_dir=target, backend_loader=lambda **kwargs: backend)


def test_entropy_means_medians_and_changes_include_all_successful_pairs():
    old, new = [], []
    for before, after in [([2,0,-1], [-1,0,2]), ([1,0,-1], [1,0,-1])]:
        for rows, logits, truncated in [(old,before,True),(new,after,False)]:
            rows.append({"status": "SUCCESS", **nli.score_result(logits, nli.softmax_logits(logits), nli.validate_label_mapping(config())),
                         "model_calls": 1, "cache_status": "MISS", "truncation": {"truncation_occurred": truncated}})
    summary = control.summarize_comparison(old, new)
    assert summary["argmax_labels_changed"] == 1
    assert summary["after"]["contradiction_predictions"] == 1
    assert summary["before"]["truncated_input_count"] == 2 and summary["after"]["truncated_input_count"] == 0
    assert summary["before"]["mean_entropy"] == pytest.approx(sum(r["normalized_entropy"] for r in old)/2)
    assert summary["after"]["median_entropy"] == pytest.approx(sum(r["normalized_entropy"] for r in new)/2)

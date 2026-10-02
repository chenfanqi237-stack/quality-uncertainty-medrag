"""Independent medical/scientific NLI feasibility probe, never a baseline replacement.

The three normalized values are NLI label scores, not calibrated probabilities.
Heavy optional dependencies are imported only when loading the real backend.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROBE_VERSION = "medical-scientific-nli-stance-v0"
MODEL_ID = "ygivenx/modernbert-base-mednli"
MODEL_REVISION = "7b0ef2489b155f8a53bdd7d56e4ecc9ad9c7b174"
HYPOTHESIS_VERSION = "question-single-option-hypothesis-v1"
PREMISE_VERSION = "pubmed-title-full-abstract-v1"
MODEL_CARD_SEQUENCE_CAP = 256
INPUT_FIELDS = ("question_stem", "candidate_option_text", "evidence_title", "evidence_abstract")
ID_FIELDS = ("question_id", "candidate_option_id", "evidence_doc_id")
SEMANTIC_LABELS = {"entailment": "SUPPORT", "contradiction": "CONTRADICT", "neutral": "IRRELEVANT"}
STANCE_ORDER = ("SUPPORT", "CONTRADICT", "IRRELEVANT")


class NLIProbeError(ValueError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(name + " must be a string" + ("" if empty else " containing text"))
    return value


def build_hypothesis(*, question_stem, candidate_option_text):
    """One fixed template; the candidate precedes the possibly long clinical stem.

    No diagnosis extraction, paraphrasing, LLM, other options or answer labels.
    The exact resulting text is saved, even when later tokenization truncates it.
    """
    stem = _text(question_stem, "question_stem").strip()
    candidate = _text(candidate_option_text, "candidate_option_text").strip()
    return "The proposed answer is: " + candidate + "\nClinical question: " + stem


def build_premise(*, evidence_title, evidence_abstract):
    title = _text(evidence_title, "evidence_title").strip()
    abstract = _text(evidence_abstract, "evidence_abstract", empty=True).strip()
    return title + ("\n\n" + abstract if abstract else "")


def build_pair(*, question_stem, candidate_option_text, evidence_title, evidence_abstract):
    return {"premise": build_premise(evidence_title=evidence_title, evidence_abstract=evidence_abstract),
            "hypothesis": build_hypothesis(question_stem=question_stem, candidate_option_text=candidate_option_text)}


def validate_label_mapping(config):
    """Inspect actual config; refuse numeric/default LABEL_0 guesses."""
    labels = getattr(config, "id2label", None)
    if getattr(config, "num_labels", None) != 3 or not isinstance(labels, dict) or len(labels) != 3:
        raise NLIProbeError("LABEL_MAPPING_FAILURE", "Downloaded config must define exactly three NLI labels")
    verified = {}
    for index, label in labels.items():
        if isinstance(index, bool) or not (type(index) is int or isinstance(index, str) and re.fullmatch(r"[0-2]", index)):
            raise NLIProbeError("LABEL_MAPPING_FAILURE", "Invalid id2label index")
        index = int(index)
        if index not in (0, 1, 2) or index in verified or not isinstance(label, str):
            raise NLIProbeError("LABEL_MAPPING_FAILURE", "Invalid or duplicate id2label entry")
        semantic = label.strip().lower()
        if semantic not in SEMANTIC_LABELS:
            raise NLIProbeError("LABEL_MAPPING_FAILURE", "Config labels lack unambiguous NLI semantics; no ID guess allowed")
        verified[index] = semantic
    if set(verified) != {0, 1, 2} or set(verified.values()) != set(SEMANTIC_LABELS):
        raise NLIProbeError("LABEL_MAPPING_FAILURE", "NLI labels must occur exactly once")
    reverse = getattr(config, "label2id", None)
    if reverse is not None:
        if not isinstance(reverse, dict) or len(reverse) != 3:
            raise NLIProbeError("LABEL_MAPPING_FAILURE", "Config label2id disagrees with id2label")
        for label, index in reverse.items():
            if not isinstance(label, str) or type(index) is not int or verified.get(index) != label.strip().lower():
                raise NLIProbeError("LABEL_MAPPING_FAILURE", "Config label2id disagrees with id2label")
    return {str(index): {"nli_label": label, "stance": SEMANTIC_LABELS[label]}
            for index, label in sorted(verified.items())}


def softmax_logits(logits):
    if not isinstance(logits, (list, tuple)) or len(logits) != 3:
        raise NLIProbeError("INVALID_MODEL_OUTPUT", "Expected exactly three raw logits")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in logits):
        raise NLIProbeError("INVALID_MODEL_OUTPUT", "Logits must be finite numbers")
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def normalized_entropy(scores):
    if len(scores) != 3 or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise NLIProbeError("INVALID_MODEL_OUTPUT", "Expected three finite normalized scores")
    if not math.isclose(math.fsum(scores), 1, rel_tol=0, abs_tol=1e-12):
        raise NLIProbeError("INVALID_MODEL_OUTPUT", "NLI label scores must sum to one")
    return -math.fsum(p * math.log(p) for p in scores if p > 0) / math.log(3)


def score_result(logits, scores, label_mapping):
    expected = softmax_logits(logits)
    entropy = normalized_entropy(scores)
    if any(not math.isclose(actual, reference, rel_tol=0, abs_tol=1e-12) for actual, reference in zip(scores, expected)):
        raise NLIProbeError("INVALID_MODEL_OUTPUT", "Scores do not match softmax of the raw logits")
    mapped = {label_mapping[str(i)]["stance"]: float(score) for i, score in enumerate(scores)}
    labels = [label_mapping[str(i)]["stance"] for i, score in enumerate(scores) if score == max(scores)]
    return {"raw_logits": list(logits), "raw_logits_by_nli_label": {label_mapping[str(i)]["nli_label"]: x for i, x in enumerate(logits)},
            "softmax_scores_in_model_label_order": list(scores),
            "nli_label_scores": mapped, "p_support": mapped["SUPPORT"], "p_contradict": mapped["CONTRADICT"],
            "p_irrelevant": mapped["IRRELEVANT"], "argmax_label": labels[0] if len(labels) == 1 else None,
            "argmax_labels": labels, "is_tied": len(labels) > 1,
            "normalized_entropy": entropy, "scores_are_calibrated_probabilities": False}


def sequence_limit(tokenizer, config, requested=MODEL_CARD_SEQUENCE_CAP):
    if type(requested) is not int or not 4 <= requested <= MODEL_CARD_SEQUENCE_CAP:
        raise ValueError("Probe sequence cap must be between 4 and the model-card cap of 256")
    raw = {"probe_model_card_cap": requested,
           "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
           "model_max_position_embeddings": getattr(config, "max_position_embeddings", None)}
    limits = [requested]
    for value in list(raw.values())[1:]:
        # Hugging Face uses enormous sentinel limits for some tokenizers.
        if type(value) is int and 0 < value < 1_000_000:
            limits.append(value)
    selected = min(limits)
    if selected <= tokenizer.num_special_tokens_to_add(pair=True):
        raise NLIProbeError("TOKENIZATION_FAILURE", "Supported pair limit leaves no room for text")
    return selected, raw


def tokenize_pair(tokenizer, config, *, premise, hypothesis, max_length=MODEL_CARD_SEQUENCE_CAP):
    limit, sources = sequence_limit(tokenizer, config, max_length)
    original = tokenizer(premise, hypothesis, add_special_tokens=True, truncation=False, padding=False)
    ids = original.get("input_ids")
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise NLIProbeError("TOKENIZATION_FAILURE", "Expected one untruncated token sequence")
    original_length = len(ids)
    encoded = tokenizer(premise, hypothesis, add_special_tokens=True,
                        truncation="longest_first", max_length=limit, padding=False, return_tensors="pt")
    shape = tuple(encoded["input_ids"].shape)
    if len(shape) != 2 or shape[0] != 1 or not 0 < shape[1] <= limit or shape[1] > original_length:
        raise NLIProbeError("TOKENIZATION_FAILURE", "Invalid final token sequence shape/length")
    if (original_length > limit) != (shape[1] < original_length):
        raise NLIProbeError("TOKENIZATION_FAILURE", "Tokenizer truncation does not match the recorded length limit")
    return encoded, {"original_token_length": original_length, "final_token_length": shape[1],
                     "truncation_occurred": shape[1] < original_length,
                     "effective_max_length": limit, "length_limit_sources": sources,
                     "truncation_strategy": "longest_first", "add_special_tokens": True, "padding": False,
                     "token_lengths_include_pair_special_tokens": True}


@dataclass(frozen=True)
class NLIModelSpec:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    tokenizer_revision: str = MODEL_REVISION
    max_length: int = MODEL_CARD_SEQUENCE_CAP
    seed: int = 42

    def __post_init__(self):
        if self.model_id != MODEL_ID:
            raise ValueError("This probe uses only the specified Hugging Face model")
        if any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{40}", v) for v in (self.model_revision, self.tokenizer_revision)):
            raise ValueError("Model and tokenizer revisions must be full immutable commit hashes")
        if self.model_revision != self.tokenizer_revision:
            raise ValueError("This probe requires model and tokenizer from the same pinned snapshot")
        if type(self.max_length) is not int or not 4 <= self.max_length <= 256 or type(self.seed) is not int:
            raise ValueError("Invalid reproducibility settings")


class TransformersNLIBackend:
    """An eval/inference-mode sequence classifier, with explicit pair truncation."""
    def __init__(self, *, tokenizer, model, torch_module, spec, snapshot_identity, device="cpu"):
        self.tokenizer, self.model, self.torch = tokenizer, model, torch_module
        self.spec, self.device = spec, device
        self.label_mapping = validate_label_mapping(model.config)
        limit, sources = sequence_limit(tokenizer, model.config, spec.max_length)
        self.identity = {"backend": "huggingface-transformers-nli", "probe_version": PROBE_VERSION,
                         **asdict(spec), "verified_label_mapping": self.label_mapping,
                         "tokenizer_settings": {"use_fast": True, "add_special_tokens": True,
                             "truncation": "longest_first", "max_length": limit, "padding": False,
                             "length_limit_sources": sources},
                         "hypothesis_version": HYPOTHESIS_VERSION, "premise_version": PREMISE_VERSION,
                         **snapshot_identity}
        if snapshot_identity.get("resolved_model_revision") != spec.model_revision or snapshot_identity.get("resolved_tokenizer_revision") != spec.tokenizer_revision:
            raise NLIProbeError("MODEL_IDENTITY_FAILURE", "Loaded snapshot revisions differ from the pinned specification")
        self.torch.manual_seed(spec.seed)
        self.torch.use_deterministic_algorithms(True)
        self.model.to(device)
        self.model.eval()

    def infer(self, *, premise, hypothesis):
        encoded, truncation = tokenize_pair(self.tokenizer, self.model.config, premise=premise,
                                            hypothesis=hypothesis, max_length=self.spec.max_length)
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
            if tuple(logits.shape) != (1, 3):
                raise NLIProbeError("INVALID_MODEL_OUTPUT", "Model must return one three-class logits vector")
            scores = self.torch.softmax(logits.to(dtype=self.torch.float64), dim=-1)
        raw = logits.detach().cpu().tolist()[0]
        normalized = scores.detach().cpu().tolist()[0]
        return {**score_result(raw, normalized, self.label_mapping), "truncation": truncation}


def load_transformers_backend(*, spec=NLIModelSpec(), hf_cache_dir, device="cpu", local_files_only=False,
                              use_existing_hf_login=False):
    """Download one immutable snapshot; never execute remote repository code."""
    try:
        import torch
        import transformers
        import huggingface_hub
        import tokenizers
        import safetensors
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise NLIProbeError("DEPENDENCIES_MISSING", "Install the isolated NLI runtime requirements first") from None
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda; the runner never starts a cloud GPU")
    if device == "cuda" and not torch.cuda.is_available():
        raise NLIProbeError("DEVICE_UNAVAILABLE", "Requested CUDA device is unavailable")
    try:
        snapshot = Path(snapshot_download(repo_id=spec.model_id, revision=spec.model_revision,
            cache_dir=str(hf_cache_dir), local_files_only=local_files_only,
            token=True if use_existing_hf_login else False,
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                            "spm.model", "model.safetensors", "README.md"]))
    except GatedRepoError:
        raise NLIProbeError("MODEL_ACCESS_REQUIRED", "Access to the pinned model was denied; no credentials or access permissions were changed") from None
    except Exception:
        raise NLIProbeError("MODEL_DOWNLOAD_FAILURE", "Pinned model snapshot could not be obtained; credentials/raw transport messages are not logged") from None
    if snapshot.name != spec.model_revision:
        raise NLIProbeError("MODEL_IDENTITY_FAILURE", "Hub snapshot directory does not identify the pinned commit")
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        if not (snapshot / required).is_file():
            raise NLIProbeError("MODEL_DOWNLOAD_FAILURE", "Pinned snapshot is missing a required model/tokenizer file")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(str(snapshot), local_files_only=True,
                                                                   use_safetensors=True, trust_remote_code=False)
    except Exception:
        raise NLIProbeError("MODEL_LOAD_FAILURE", "Pinned safe model/tokenizer could not be loaded") from None
    if not getattr(tokenizer, "is_fast", False):
        raise NLIProbeError("TOKENIZATION_FAILURE", "Expected the pinned fast tokenizer")
    identity = {"resolved_model_revision": snapshot.name, "resolved_tokenizer_revision": snapshot.name,
                "raw_id2label": {str(k): v for k, v in model.config.id2label.items()},
                "raw_label2id": model.config.label2id,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "snapshot_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(snapshot.iterdir()) if p.is_file()},
                "runtime_versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "huggingface_hub": huggingface_hub.__version__, "tokenizers": tokenizers.__version__,
                    "safetensors": safetensors.__version__},
                "model_dtype": str(model.dtype), "logit_softmax_dtype": "torch.float64",
                "device": device, "model_training": False, "deterministic_algorithms": True,
                "trust_remote_code": False, "use_safetensors": True,
                "hub_authentication_mode": "existing_login" if use_existing_hf_login else "anonymous"}
    return TransformersNLIBackend(tokenizer=tokenizer, model=model, torch_module=torch, spec=spec,
                                   snapshot_identity=identity, device=device)


def build_cache_context(*, ids, inputs, pair, backend_identity):
    if set(ids) != set(ID_FIELDS) or set(inputs) != set(INPUT_FIELDS) or set(pair) != {"premise", "hypothesis"}:
        raise ValueError("Only judgment IDs and the four isolated text fields are allowed")
    for name, value in ids.items():
        _text(value, name)
    for name, value in inputs.items():
        _text(value, name, empty=name == "evidence_abstract")
    expected = build_pair(**inputs)
    if pair != expected:
        raise ValueError("Pair differs from the fixed deterministic construction")
    return {**ids, "probe_version": PROBE_VERSION, "hypothesis_version": HYPOTHESIS_VERSION,
            "premise_version": PREMISE_VERSION,
            "input_sha256": {name: text_hash(value) for name, value in inputs.items()},
            "pair_sha256": {name: text_hash(value) for name, value in pair.items()},
            "backend_identity": backend_identity}


class NLIStanceClassifier:
    """Standalone cache for the probe; never writes Qwen v1/logprob-v0 caches."""
    def __init__(self, backend, *, cache_dir):
        self.backend, self.cache_dir = backend, Path(cache_dir)
        if {"v1", "logprob_v0"}.intersection(self.cache_dir.parts):
            raise ValueError("NLI must have an independent cache directory")

    def classify_texts(self, *, question_id, candidate_option_id, evidence_doc_id,
                       question_stem, candidate_option_text, evidence_title, evidence_abstract):
        ids = dict(question_id=question_id, candidate_option_id=candidate_option_id, evidence_doc_id=evidence_doc_id)
        inputs = dict(question_stem=question_stem, candidate_option_text=candidate_option_text,
                      evidence_title=evidence_title, evidence_abstract=evidence_abstract)
        pair = build_pair(**inputs)
        context = build_cache_context(ids=ids, inputs=inputs, pair=pair, backend_identity=self.backend.identity)
        key = text_hash(canonical(context))
        path = self.cache_dir / (key + ".json")
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("context") != context or entry.get("cache_key") != key or entry.get("integrity_sha256") != text_hash(canonical({k: v for k, v in entry.items() if k != "integrity_sha256"})):
                raise NLIProbeError("CACHE_INTEGRITY_FAILURE", "Existing NLI cache differs; refusing overwrite or regeneration")
            result = entry["result"]
            if result["status"] == "SUCCESS":
                score_result(result["raw_logits"], result["softmax_scores_in_model_label_order"], self.backend.label_mapping)
            return {**result, **ids, "cache_key": key, "cache_status": "HIT", "model_calls": 0}
        try:
            result = {"status": "SUCCESS", **self.backend.infer(**pair)}
        except NLIProbeError as exc:
            result = {"status": exc.status, "error": str(exc)}
        result.update(pair=pair, input_sha256=context["input_sha256"], pair_sha256=context["pair_sha256"],
                      model_metadata=self.backend.identity, hypothesis_version=HYPOTHESIS_VERSION)
        entry = {"cache_key": key, "context": context, "result": result,
                 "created_at": datetime.now(timezone.utc).isoformat(), "model_calls": 1}
        entry["integrity_sha256"] = text_hash(canonical(entry))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical(entry) + "\n")
        return {**result, **ids, "cache_key": key, "cache_status": "MISS", "model_calls": 1}

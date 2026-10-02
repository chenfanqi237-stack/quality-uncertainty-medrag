"""Experimental sentence-level NLI; independent of retrieval and aggregation."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from datetime import datetime, timezone

from . import nli_stance as nli

VERSION = "modernbert-mednli-sentence-v1"
SPLITTER_VERSION = "pubmed-punctuation-sentences-v1"
SELECTION_VERSION = "maximum-support-or-contradict-earliest-index-v1"
CACHE_NAMESPACE = "nli_sentence_v1"
# Fixed general rules, never customized to a development question.
ABBREVIATIONS = frozenset({"dr.", "mr.", "mrs.", "ms.", "prof.", "vs.", "fig.", "figs.",
    "ref.", "refs.", "no.", "nos.", "al.", "e.g.", "i.e.", "etc.", "p.o.", "i.v.", "approx."})
BOUNDARY = re.compile(r'''[.!?]+["'”’\)\]]*(?=\s|$)''')


def split_abstract(abstract):
    """Punctuation boundaries, protecting decimals, abbreviations and initials.

    Preserve exact within-sentence text; trim only surrounding whitespace.
    Text without terminal punctuation remains a sentence. No external model.
    """
    nli._text(abstract, "evidence_abstract", empty=True)
    sentences, start = [], 0
    for match in BOUNDARY.finditer(abstract):
        punctuation = match.group().rstrip('"\'”’) ]')
        if punctuation == ".":
            prefix = abstract[:match.start() + 1]
            token = re.search(r"(\S+)$", prefix)
            token = token.group(1) if token else ""
            token = token.lstrip('(["\'“‘')
            if token.lower() in ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.){2,}|[A-Z]\.", token):
                continue
        text = abstract[start:match.end()].strip()
        if text:
            sentences.append(text)
        start = match.end()
    tail = abstract[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def evidence_sentences(*, title, abstract):
    title = nli._text(title, "evidence_title").strip()
    sentences = [{"sentence_index": 0, "source": "title", "abstract_sentence_index": None, "text": title}]
    sentences.extend({"sentence_index": index + 1, "source": "abstract", "abstract_sentence_index": index, "text": text}
                     for index, text in enumerate(split_abstract(abstract)))
    return sentences


def directional_strength(scores):
    if set(scores) != set(nli.STANCE_ORDER):
        raise nli.NLIProbeError("INVALID_MODEL_OUTPUT", "Expected exactly S/C/I sentence scores")
    nli.normalized_entropy([scores[name] for name in nli.STANCE_ORDER])
    return max(scores["SUPPORT"], scores["CONTRADICT"])


def select_sentence(sentences):
    if not sentences or any(row["status"] != "SUCCESS" for row in sentences):
        raise nli.NLIProbeError("SENTENCE_INFERENCE_FAILURE", "Every sentence requires a valid output before selection")
    indices = [row["sentence_index"] for row in sentences]
    if indices != list(range(len(sentences))):
        raise ValueError("Sentence indices must preserve title-first source ordering")
    # Exact ties retain the earliest source index; no directional label filtering.
    return max(sentences, key=lambda row: directional_strength(row["nli_label_scores"]))


class SentenceBackend:
    """Delegate unchanged softmax/entropy to the supported-context backend."""
    def __init__(self, backend):
        self.backend = backend
        self.tokenizer = backend.tokenizer
        self.label_mapping = backend.label_mapping
        self.identity = copy.deepcopy(backend.identity)
        self.identity.update(probe_version=VERSION, experiment_version=VERSION,
                             sentence_split_version=SPLITTER_VERSION, sentence_selection_version=SELECTION_VERSION)

    def infer(self, *, premise, hypothesis):
        return self.backend.infer(premise=premise, hypothesis=hypothesis)


def cache_context(*, ids, inputs, sentences, backend_identity):
    if set(ids) != set(nli.ID_FIELDS) or set(inputs) != set(nli.INPUT_FIELDS):
        raise ValueError("Only judgment IDs and four isolated source texts are allowed")
    pair = nli.build_pair(**inputs)
    expected = evidence_sentences(title=inputs["evidence_title"], abstract=inputs["evidence_abstract"])
    if sentences != expected:
        raise ValueError("Sentence sequence differs from the deterministic splitter")
    return {**ids, "classifier_version": VERSION, "sentence_split_version": SPLITTER_VERSION,
            "sentence_selection_version": SELECTION_VERSION, "hypothesis_version": nli.HYPOTHESIS_VERSION,
            "input_sha256": {name: nli.text_hash(value) for name, value in inputs.items()},
            "hypothesis_sha256": nli.text_hash(pair["hypothesis"]),
            "sentences": [{"sentence_index": s["sentence_index"], "source": s["source"],
                           "text_sha256": nli.text_hash(s["text"])} for s in sentences],
            "backend_identity": backend_identity,
            "effective_max_length": backend_identity["tokenizer_settings"]["max_length"]}


def sentence_token_length(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, padding=False)
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or any(type(value) is not int for value in ids):
        raise nli.NLIProbeError("TOKENIZATION_FAILURE", "Expected untruncated sentence token IDs")
    return len(ids)


class SentenceNLIClassifier:
    """Cache every sentence plus the selected full evidence distribution."""
    def __init__(self, backend, *, cache_dir):
        self.backend = SentenceBackend(backend)
        self.cache_dir = Path(cache_dir)
        if self.cache_dir.name != CACHE_NAMESPACE:
            raise ValueError("Sentence NLI must use the dedicated new cache namespace")
        self.sentence_classifier = nli.NLIStanceClassifier(self.backend, cache_dir=self.cache_dir / "sentences")

    def classify_texts(self, *, question_id, candidate_option_id, evidence_doc_id,
                       question_stem, candidate_option_text, evidence_title, evidence_abstract):
        ids = dict(question_id=question_id, candidate_option_id=candidate_option_id, evidence_doc_id=evidence_doc_id)
        inputs = dict(question_stem=question_stem, candidate_option_text=candidate_option_text,
                      evidence_title=evidence_title, evidence_abstract=evidence_abstract)
        sentences = evidence_sentences(title=evidence_title, abstract=evidence_abstract)
        context = cache_context(ids=ids, inputs=inputs, sentences=sentences, backend_identity=self.backend.identity)
        key = nli.text_hash(nli.canonical(context))
        path = self.cache_dir / "evidence" / (key + ".json")
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("context") != context or entry.get("cache_key") != key or entry.get("integrity_sha256") != nli.text_hash(nli.canonical({k:v for k,v in entry.items() if k != "integrity_sha256"})):
                raise nli.NLIProbeError("CACHE_INTEGRITY_FAILURE", "Existing sentence NLI cache differs; refusing overwrite")
            result = entry["result"]
            self._validate_result(result)
            return {**result, **ids, "cache_key": key, "cache_status": "HIT", "model_calls": 0}
        judged = []
        for sentence in sentences:
            prediction = self.sentence_classifier.classify_texts(**ids,
                question_stem=question_stem, candidate_option_text=candidate_option_text,
                evidence_title=sentence["text"], evidence_abstract="")
            item = {**sentence, **prediction,
                    "sentence_token_length": sentence_token_length(self.backend.tokenizer, sentence["text"]),
                    "sentence_token_length_includes_special_tokens": False}
            if item["status"] == "SUCCESS":
                item["directional_strength"] = directional_strength(item["nli_label_scores"])
            judged.append(item)
        result = {"classifier_version": VERSION, "sentences": judged,
                  "sentence_count": len(judged), "input_sha256": context["input_sha256"],
                  "hypothesis": nli.build_hypothesis(question_stem=question_stem, candidate_option_text=candidate_option_text),
                  "model_metadata": self.backend.identity,
                  "hypothesis_version": nli.HYPOTHESIS_VERSION,
                  "model_calls": sum(s["model_calls"] for s in judged)}
        try:
            selected = select_sentence(judged)
            result.update(status="SUCCESS", selected_sentence_index=selected["sentence_index"],
                selected_sentence=selected["text"], selected_source=selected["source"],
                directional_strength=selected["directional_strength"],
                **{name:selected[name] for name in ("raw_logits", "raw_logits_by_nli_label", "softmax_scores_in_model_label_order",
                   "nli_label_scores", "p_support", "p_contradict", "p_irrelevant", "argmax_label", "argmax_labels",
                   "is_tied", "normalized_entropy", "scores_are_calibrated_probabilities")})
        except nli.NLIProbeError as exc:
            result.update(status=exc.status, error=str(exc))
        self._validate_result(result)
        entry = {"cache_key": key, "context": context, "result": result,
                 "created_at": datetime.now(timezone.utc).isoformat()}
        entry["integrity_sha256"] = nli.text_hash(nli.canonical(entry))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(nli.canonical(entry) + "\n")
        return {**result, **ids, "cache_key": key, "cache_status": "MISS"}

    def _validate_result(self, result):
        for sentence in result["sentences"]:
            if sentence["status"] == "SUCCESS":
                nli.score_result(sentence["raw_logits"], sentence["softmax_scores_in_model_label_order"], self.backend.label_mapping)
        if result["status"] == "SUCCESS":
            selected = select_sentence(result["sentences"])
            if selected["sentence_index"] != result["selected_sentence_index"] or selected["nli_label_scores"] != result["nli_label_scores"] or selected["argmax_label"] != result["argmax_label"]:
                raise nli.NLIProbeError("CACHE_INTEGRITY_FAILURE", "Evidence scores do not equal the selected sentence's full distribution")

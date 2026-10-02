import hashlib
import json
from pathlib import Path

import pytest

from quality_uncertainty_medrag import nli_stance as nli
from quality_uncertainty_medrag import stance_nli_probe as probe


class Backend:
    label_mapping = {"0": {"nli_label": "neutral", "stance": "IRRELEVANT"},
                     "1": {"nli_label": "contradiction", "stance": "CONTRADICT"},
                     "2": {"nli_label": "entailment", "stance": "SUPPORT"}}
    identity = {"model_id": nli.MODEL_ID, "model_revision": nli.MODEL_REVISION,
                "resolved_model_revision": nli.MODEL_REVISION, "tokenizer_revision": nli.MODEL_REVISION,
                "resolved_tokenizer_revision": nli.MODEL_REVISION, "hypothesis_version": nli.HYPOTHESIS_VERSION,
                "tokenizer_settings": {"max_length": 256, "truncation": "longest_first"},
                "raw_id2label": {"0": "neutral", "1": "contradiction", "2": "entailment"},
                "raw_label2id": {"neutral": 0, "contradiction": 1, "entailment": 2}, "parameter_count": 150}

    def __init__(self):
        self.calls = []

    def infer(self, **pair):
        self.calls.append(pair)
        return {**nli.score_result([-.5, -1, 2], nli.softmax_logits([-.5, -1, 2]), self.label_mapping),
                "truncation": {"original_token_length": 320, "final_token_length": 256, "truncation_occurred": True}}


@pytest.fixture
def diagnostic(tmp_path, monkeypatch):
    root = tmp_path / "project"
    source = root / probe.SOURCE_RELATIVE
    source.mkdir(parents=True)
    inputs, old = [], []
    for index, (qid, option, pmid) in enumerate(probe.DIAGNOSTIC_KEYS):
        row = dict(question_id=qid, candidate_option_id=option, evidence_doc_id=pmid,
                   question_stem=f"Synthetic stem {index}", candidate_option_text=f"Only candidate {index}",
                   evidence_title=f"Title {index}", evidence_abstract=f"Full abstract {index}")
        inputs.append(row)
        old.append({**{field: row[field] for field in nli.ID_FIELDS}, "classifier_version": "llm-medical-stance-v1",
                    "status": "SUCCESS", "input_sha256": {field: nli.text_hash(row[field]) for field in nli.INPUT_FIELDS},
                    "p_support": .9, "p_contradict": 0., "p_irrelevant": .1,
                    "argmax_label": "SUPPORT", "cache_key": f"existing-{index}"})
    data = "".join(nli.canonical(row) + "\n" for row in inputs).encode()
    (source / "inputs.jsonl").write_bytes(data)
    probe.save_jsonl(source / "stances.jsonl", old)
    monkeypatch.setattr(probe, "DIAGNOSTIC_INPUT_SHA256", hashlib.sha256(data).hexdigest())
    return root, source, inputs


def test_exact_eighteen_pairs_and_original_reference_order(diagnostic):
    root, source, inputs = diagnostic
    rows, references, hashes = probe.load_diagnostic(source)
    assert len(rows) == len(references) == 18
    assert tuple(probe.key(r) for r in rows) == probe.DIAGNOSTIC_KEYS
    assert rows == inputs
    assert hashes["input_sha256"] == hashlib.sha256((source / "inputs.jsonl").read_bytes()).hexdigest()
    assert all(int(row["question_id"].rsplit("-", 1)[1]) <= 30 for row in rows)


def test_changed_source_bytes_refused_without_resampling(diagnostic):
    root, source, _ = diagnostic
    path = source / "inputs.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input bytes differ"):
        probe.load_diagnostic(source)


def test_order_changes_are_refused_even_with_matching_hash(diagnostic, monkeypatch):
    root, source, inputs = diagnostic
    probe.save_jsonl(source / "inputs.jsonl", list(reversed(inputs)))
    monkeypatch.setattr(probe, "DIAGNOSTIC_INPUT_SHA256", hashlib.sha256((source / "inputs.jsonl").read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="fixed eighteen"):
        probe.load_diagnostic(source)


@pytest.mark.parametrize("field", ["answer", "answer_idx", "options", "metadata", "candidate_correctness"])
def test_extra_gold_other_options_or_metadata_rejected(diagnostic, monkeypatch, field):
    root, source, inputs = diagnostic
    inputs[0][field] = "SECRET_OTHER_OPTIONS_OR_GOLD"
    probe.save_jsonl(source / "inputs.jsonl", inputs)
    monkeypatch.setattr(probe, "DIAGNOSTIC_INPUT_SHA256", hashlib.sha256((source / "inputs.jsonl").read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="Unexpected field"):
        probe.load_diagnostic(source)


def test_reference_must_describe_the_same_four_texts(diagnostic):
    root, source, _ = diagnostic
    old = [json.loads(line) for line in (source / "stances.jsonl").read_text().splitlines()]
    old[0]["input_sha256"]["candidate_option_text"] = "mismatched"
    probe.save_jsonl(source / "stances.jsonl", old)
    with pytest.raises(ValueError, match="exact same four texts"):
        probe.load_diagnostic(source)


def test_prepare_only_saves_exact_hypotheses_without_network_model_or_caches(diagnostic):
    root, source, inputs = diagnostic
    def forbidden_loader(**kwargs):
        raise AssertionError("No loading or inference allowed")
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    report, directory = probe.run_probe(root=root, prepare_only=True, backend_loader=forbidden_loader)
    assert report["status"] == "PREPARED_NOT_EXECUTED"
    assert report["model_calls"] == 0 and report["actual_model_revision_used"] is None
    assert report["verified_label_mapping"] is None
    pairs = [json.loads(line) for line in (directory / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == 18
    for row, pair in zip(inputs, pairs):
        assert pair["hypothesis"] == nli.build_hypothesis(question_stem=row["question_stem"], candidate_option_text=row["candidate_option_text"])
        assert pair["premise"] == row["evidence_title"] + "\n\n" + row["evidence_abstract"]
    assert not (root / "data/cache/stance/nli_v0").exists()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original


def test_gated_model_blocks_without_fabricating_results(diagnostic):
    root, source, inputs = diagnostic
    def denied(**kwargs):
        raise nli.NLIProbeError("MODEL_ACCESS_REQUIRED", "Approved gated-model access is required")
    report, directory = probe.run_probe(root=root, backend_loader=denied)
    assert report["status"] == "MODEL_ACCESS_REQUIRED"
    assert report["results"] == [] and report["model_calls"] == 0
    assert report["actual_model_revision_used"] is None
    assert not (directory / "results.jsonl").exists()
    assert not (root / "data/cache/stance/nli_v0").exists()
    text = (directory / "comparison.md").read_text()
    assert "NOT YET VERIFIED" in text
    assert text.count("| medqa-us-dev-") == 18


def test_full_run_is_isolated_and_cache_reuse_leaves_outputs_byte_identical(diagnostic):
    root, source, inputs = diagnostic
    backend = Backend()
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    first, directory = probe.run_probe(root=root, backend_loader=lambda **kwargs: backend)
    assert first["status"] == "COMPLETE"
    assert first["summary"]["valid_nli_outputs"] == 18
    assert first["summary"]["actual_model_calls"] == 18
    assert first["summary"]["qwen_v1_hard_label_agreement_count"] == 18
    assert first["summary"]["truncated_input_count"] == 18
    assert first["summary"]["truncated_input_percentage"] == 100
    assert len(backend.calls) == 18
    for pair, row in zip(backend.calls, inputs):
        assert set(pair) == {"premise", "hypothesis"}
        assert pair == nli.build_pair(**{field: row[field] for field in nli.INPUT_FIELDS})
        assert "SECRET_OTHER_OPTIONS_OR_GOLD" not in nli.canonical(pair)
    first_outputs = {p.name: p.read_bytes() for p in directory.iterdir()}
    caches = root / "data/cache/stance/nli_v0"
    assert len(list(caches.glob("*.json"))) == 18
    first_cache = {p.name: p.read_bytes() for p in caches.iterdir()}
    second, second_dir = probe.run_probe(root=root, backend_loader=lambda **kwargs: backend)
    assert directory != second_dir and second["summary"]["actual_model_calls"] == 0
    assert second["summary"]["cache_hits"] == 18 and len(backend.calls) == 18
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == first_outputs
    assert {p.name: p.read_bytes() for p in caches.iterdir()} == first_cache
    audits = json.loads((directory / "request_audit.json").read_text())
    assert len(audits) == 18
    assert all(a["model_input_fields"] == ["premise", "hypothesis"] and a["gold_or_other_options_exposed"] is False for a in audits)


def test_existing_output_and_old_qwen_output_paths_are_refused(diagnostic):
    root, source, inputs = diagnostic
    with pytest.raises(ValueError, match="new diagnostic_18"):
        probe.run_probe(root=root, output_dir=source, prepare_only=True)
    directory = root / "outputs/stance_nli_probe/dev_1_30/modernbert_mednli_diagnostic_18/already-exists"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="overwrite"):
        probe.run_probe(root=root, output_dir=directory, prepare_only=True)


def test_entropy_buckets_are_report_only_not_decision_thresholds():
    rows = []
    for entropy in (0, .049, .05, .249, .25, .50, .501, 1):
        rows.append({"status": "SUCCESS", "argmax_label": "CONTRADICT", "nli_label_scores": dict(zip(nli.STANCE_ORDER, (.2, .6, .2))),
                     "normalized_entropy": entropy, "hard_label_agreement": False,
                     "truncation": {"truncation_occurred": False}, "model_calls": 1, "cache_status": "MISS"})
    summary = probe.summarize(rows)
    assert summary["entropy_buckets"] == {"<0.05": 2, "0.05-0.25": 2, "0.25-0.50": 2, ">0.50": 2}
    assert summary["contradiction_predictions"] == 8
    assert summary["qwen_v1_hard_label_agreement_count"] == 0
    assert summary["accuracy_evaluated"] is False and summary["superiority_claimed"] is False

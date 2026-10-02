import copy
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from quality_uncertainty_medrag import llm_stance_logprob as probe
from quality_uncertainty_medrag.llm_stance import StanceModelConfig


def token(text, lp=-0.1, alternatives=None):
    out = {"token": text, "bytes": list(text.encode()), "logprob": lp}
    if alternatives is not None:
        out["top_logprobs"] = alternatives
    return out


def response(code="1", *, lps=(-0.1, -2., -3.)):
    top = [token(c, lp) for c, lp in zip("123", lps)]
    return {"model": "qwen3:8b", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": code, "thinking": "private 1 2 3"},
            "logprobs": [token("private 1 2 3"), token("</think>"), token("\n\n"),
                         token(code, lps[int(code)-1], top)]}


INPUTS = dict(question_stem="A stem", candidate_option_text="ONE selected candidate",
              evidence_title="A real title", evidence_abstract="A real abstract")
IDS = dict(question_id="medqa-us-dev-000004", candidate_option_id="A", evidence_doc_id="40894990")


@pytest.mark.parametrize("content", ["1", "2", "3", " 1\n", "\n3\t"])
def test_final_code_valid(content):
    assert probe.validate_class_code(content) == content.strip()


@pytest.mark.parametrize("content", [None, 1, "", "0", "4", "1.", "SUPPORT", "1 = SUPPORT", "1 2", "```1```", "<think>x</think>1"])
def test_final_code_invalid(content):
    with pytest.raises(probe.LogprobProbeError) as exc:
        probe.validate_class_code(content)
    assert exc.value.status == "INVALID_FINAL_CONTENT"


def test_alignment_uses_final_token_not_hidden_digits():
    raw = response("2")
    raw["logprobs"][0] = token("private 2 and 2 again")
    result = probe.parse_logprob_response(raw)
    assert result["alignment"]["final_token_index"] == 3
    assert result["alignment"]["class_token_count"] == 1
    assert result["final_class_code"] == "2"
    assert "private" not in json.dumps(result)
    assert result["all_label_tokens_present"]


def test_split_boundary_tokens_and_visible_whitespace():
    raw = response()
    raw["message"]["content"] = "\n1\n"
    raw["logprobs"] = [token("private"), token("</th"), token("ink>"), token("\n"), raw["logprobs"][-1], token("\n")]
    assert probe.align_final_class_token(raw)[2]["final_token_index"] == 4


def test_no_thinking_content_only_exact_stream():
    raw = response()
    raw["message"].pop("thinking")
    raw["logprobs"] = raw["logprobs"][-1:]
    assert probe.parse_logprob_response(raw)["alignment"]["strategy"] == "NO_THINKING_EXACT_VISIBLE_STREAM"


@pytest.mark.parametrize("change", ["unmarked", "thinking_content_only", "suffix_extra", "merged_space", "multiple_close", "no_logprobs", "bytes_mismatch", "token_text_mismatch"])
def test_alignment_fails_explicitly(change):
    raw = response()
    if change == "unmarked": raw["logprobs"].pop(1)
    if change == "thinking_content_only": raw["logprobs"] = raw["logprobs"][-1:]
    if change == "suffix_extra": raw["logprobs"].append(token("."))
    if change == "merged_space": raw["message"]["content"] = " 1"; raw["logprobs"] = [token("</think>"), token(" 1")]
    if change == "multiple_close": raw["logprobs"].insert(0, token("</think>"))
    if change == "no_logprobs": raw.pop("logprobs")
    if change == "bytes_mismatch": raw["logprobs"][-1]["bytes"] = [50]
    if change == "token_text_mismatch": raw["logprobs"][-1]["token"] = "wrong"
    with pytest.raises(probe.LogprobProbeError) as exc:
        probe.parse_logprob_response(raw)
    assert exc.value.status == "FINAL_TOKEN_ALIGNMENT_FAILURE"


@pytest.mark.parametrize("code", "123")
def test_missing_class_token_is_incomplete_not_zero(code):
    raw = response()
    raw["logprobs"][-1]["top_logprobs"] = [t for t in raw["logprobs"][-1]["top_logprobs"] if t["token"] != code]
    with pytest.raises(probe.LogprobProbeError) as exc:
        probe.parse_logprob_response(raw)
    assert exc.value.status == "LOGPROB_INCOMPLETE"
    assert exc.value.diagnostics["missing_label_tokens"] == [code]
    assert code not in exc.value.diagnostics["raw_label_logprobs"]


def test_whitespace_prefixed_alternative_is_not_exact_label_token():
    raw = response()
    raw["logprobs"][-1]["top_logprobs"][1] = token(" 2", -2)
    with pytest.raises(probe.LogprobProbeError, match="absent"):
        probe.parse_logprob_response(raw)


@pytest.mark.parametrize("lps", [(-10000., -10001., -10002.), (-.1, -2., -3.), (0., 0., 0.), (0., -1000., -1000.)])
def test_stable_normalization_sum_and_entropy(lps):
    actual = probe.normalized_label_likelihoods(dict(zip("123", lps)))
    assert math.fsum(actual.values()) == pytest.approx(1, abs=1e-15)
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in actual.values())
    assert 0 <= probe.normalized_entropy(actual) <= 1 + 1e-15
    shifted = probe.normalized_label_likelihoods(dict(zip("123", [v - 1000 for v in lps])))
    assert actual == pytest.approx(shifted)


def test_uniform_and_one_hot_entropy():
    assert probe.normalized_entropy(dict(SUPPORT=1/3, CONTRADICT=1/3, IRRELEVANT=1/3)) == pytest.approx(1)
    assert probe.normalized_entropy(dict(SUPPORT=1., CONTRADICT=0., IRRELEVANT=0.)) == 0


@pytest.mark.parametrize("lp", [float("nan"), float("inf"), -float("inf"), True, "-1", .01])
def test_invalid_logprob_fails(lp):
    with pytest.raises(probe.LogprobProbeError):
        probe.normalized_label_likelihoods({"1": lp, "2": -2, "3": -3})


def test_inconsistent_selected_logprob_and_duplicates_fail():
    raw = response(); raw["logprobs"][-1]["logprob"] = -.2
    with pytest.raises(probe.LogprobProbeError, match="disagrees"):
        probe.parse_logprob_response(raw)
    raw = response(); raw["logprobs"][-1]["top_logprobs"].append(token("2", -3))
    with pytest.raises(probe.LogprobProbeError, match="Duplicate"):
        probe.parse_logprob_response(raw)


def test_tied_label_likelihoods_have_explicit_unresolved_argmax():
    result = probe.parse_logprob_response(response(lps=(-1, -1, -1)))
    assert result["argmax_label"] is None
    assert result["final_class_code"] == "1"


def test_native_endpoint_settings_and_only_one_candidate(monkeypatch):
    calls = []
    def opened(request, *, timeout):
        body = json.loads(request.data)
        calls.append(body)
        assert request.full_url == "http://localhost:11434/api/chat"
        assert request.get_method() == "POST"
        assert body["stream"] is False and body["think"] is True
        assert body["logprobs"] is True and body["top_logprobs"] == 20
        assert body["options"] == {"temperature": 0., "seed": 42}
        assert len(body["messages"]) == 1
        prompt = body["messages"][0]["content"]
        model_input = json.loads(prompt.split("Input JSON:\n", 1)[1])
        assert model_input == INPUTS
        assert set(model_input) == set(probe.MODEL_INPUT_FIELDS)
        return io.StringIO(json.dumps(response()))
    monkeypatch.setattr(probe, "urlopen", opened)
    backend = probe.NativeOllamaLogprobBackend()
    assert backend.judge(INPUTS)["status"] == "SUCCESS"
    assert len(calls) == 1
    for field in ("answer", "answer_idx", "gold_label", "options", "upstream"):
        with pytest.raises(ValueError): backend.judge({**INPUTS, field: "FORBIDDEN"})
    assert len(calls) == 1


def test_exact_cache_reuse_and_no_thinking_saved(tmp_path, monkeypatch):
    calls = []
    def opened(request, *, timeout):
        calls.append(request)
        return io.StringIO(json.dumps(response()))
    monkeypatch.setattr(probe, "urlopen", opened)
    classifier = probe.LogprobStanceClassifier(probe.NativeOllamaLogprobBackend(), cache_dir=tmp_path / "logprob_v0")
    first = classifier.classify_texts(**IDS, **INPUTS)
    second = classifier.classify_texts(**IDS, **INPUTS)
    assert first["normalized_label_likelihoods"] == second["normalized_label_likelihoods"]
    assert first["cache_status"] == "MISS" and second["cache_status"] == "HIT"
    assert second["generation_attempts"] == 0 and len(calls) == 1
    saved = (tmp_path / "logprob_v0" / (first["cache_key"] + ".json")).read_text()
    assert "private" not in saved and '"thinking": "' not in saved
    assert first["cache_key"] == second["cache_key"]


def test_incomplete_is_cached_and_never_regenerated(tmp_path, monkeypatch):
    raw = response(); raw["logprobs"][-1]["top_logprobs"].pop()
    calls = []
    def opened(request, *, timeout):
        calls.append(request)
        return io.StringIO(json.dumps(raw))
    monkeypatch.setattr(probe, "urlopen", opened)
    classifier = probe.LogprobStanceClassifier(probe.NativeOllamaLogprobBackend(), cache_dir=tmp_path / "logprob_v0")
    result = classifier.classify_texts(**IDS, **INPUTS)
    assert result["status"] == "LOGPROB_INCOMPLETE"
    assert result["normalized_label_likelihoods"] is None
    assert classifier.classify_texts(**IDS, **INPUTS)["cache_status"] == "HIT"
    assert len(calls) == 1


def test_forbid_v1_cache_and_corrupt_entry(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        probe.LogprobStanceClassifier(probe.NativeOllamaLogprobBackend(), cache_dir=tmp_path / "v1")
    monkeypatch.setattr(probe, "urlopen", lambda *a, **kw: io.StringIO(json.dumps(response())))
    classifier = probe.LogprobStanceClassifier(probe.NativeOllamaLogprobBackend(), cache_dir=tmp_path / "logprob_v0")
    result = classifier.classify_texts(**IDS, **INPUTS)
    path = tmp_path / "logprob_v0" / (result["cache_key"] + ".json")
    entry = json.loads(path.read_text()); entry["result"]["argmax_label"] = "CONTRADICT"
    path.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="checksum"):
        classifier.classify_texts(**IDS, **INPUTS)

import io
import json
from pathlib import Path

import pytest

from quality_uncertainty_medrag import ollama_backend, stance_smoke
from quality_uncertainty_medrag.llm_stance import StanceModelConfig


def inputs():
    return [{"question_id": f"medqa-us-dev-{index:06d}", "candidate_option_id": label,
             "evidence_doc_id": str(100 + index), "question_stem": f"Stem {index}",
             "candidate_option_text": "Candidate " + label, "evidence_title": "Title", "evidence_abstract": "Abstract"}
            for index in (11,12,13) for label in ("A","B")]


def identity():
    cfg = StanceModelConfig()
    return {"model_name": cfg.model, "backend_id": cfg.backend_id,
            "model_digest": cfg.model_digest, "ollama_version": cfg.ollama_version}


def test_prepare_uses_first_options_and_first_rank_without_gold(tmp_path):
    questions = [{"id": f"medqa-us-dev-{i:06d}","question": f"Stem {i}",
        "options": {"B":"first", "A":"second", "C":"OTHER_OPTION_SECRET"},
        "answer":"GOLD_SECRET","answer_idx":2,"metadata":{"upstream":{"answer":"GOLD_SECRET"}}} for i in (11,12,13,31)]
    evidence = [{"question_id": f"medqa-us-dev-{i:06d}","doc_id":str(100*i+r), "rank":r,
        "metadata":{"title":f"Title {r}","abstract":"" if r==1 else f"Abstract {r}","query":"IGNORED_QUERY"}}
        for i in (11,12,13) for r in (3,2,1)]
    qpath, epath = tmp_path/"q.jsonl",tmp_path/"e.jsonl"
    qpath.write_text("\n".join(json.dumps(x) for x in questions),encoding="utf-8")
    epath.write_text("\n".join(json.dumps(x) for x in evidence),encoding="utf-8")
    rows = stance_smoke.prepare_development_inputs(qpath,epath)
    assert len(rows)==6 and [x["candidate_option_id"] for x in rows]==["B","A"]*3
    assert all(x["evidence_title"]=="Title 2" for x in rows)
    text = json.dumps(rows)
    assert all(secret not in text for secret in ("OTHER_OPTION_SECRET","GOLD_SECRET","IGNORED_QUERY","000031"))
    assert all(set(x)==stance_smoke.RECORD_FIELDS for x in rows)


@pytest.mark.parametrize("ids",[("medqa-us-dev-000031","medqa-us-dev-000032"),
    ("medqa-us-dev-000000","medqa-us-dev-000001"),("medqa-us-dev-000011",),
    ("medqa-us-dev-000011",)*2,tuple(f"medqa-us-dev-{i:06d}" for i in range(1,5))])
def test_only_two_or_three_development_questions_allowed(ids):
    with pytest.raises(ValueError):
        stance_smoke._development_ids(ids)


@pytest.mark.parametrize("field",["answer","answer_idx","metadata","options","gold_label"])
def test_input_bundle_rejects_forbidden_extra_fields(field):
    rows = inputs()
    rows[0][field] = "secret"
    with pytest.raises(ValueError):
        stance_smoke.validate_smoke_inputs(rows)


def test_duplicate_judgments_rejected():
    rows=inputs()
    rows[1]=dict(rows[0])
    with pytest.raises(ValueError):
        stance_smoke.validate_smoke_inputs(rows)


@pytest.mark.parametrize("field,value",[("model_name","other"),("model_digest","a"*64),
    ("ollama_version","0.34.5"),("backend_id","other")])
def test_smoke_refuses_model_identity_drift(field,value):
    with pytest.raises(ValueError):
        stance_smoke._required_identity({**identity(),field:value})


def test_mocked_actual_ollama_payloads_and_cache_reuse(tmp_path,monkeypatch):
    ipath=tmp_path/"inputs.jsonl"
    rows=inputs()
    ipath.write_text("\n".join(json.dumps(x) for x in rows),encoding="utf-8")
    monkeypatch.setattr(stance_smoke,"verify_freeze",lambda *args:{"frozen":"digest"})
    monkeypatch.setattr(stance_smoke,"gpu_preflight",lambda *args:{"identity":identity(),"gpu_info":"MOCK GPU","model_load_requests":0})
    monkeypatch.setattr(stance_smoke,"ollama_identity",lambda *args:identity())
    captured=[]
    def transport(request,*,timeout):
        body=json.loads(request.data)
        captured.append(body)
        return io.BytesIO(json.dumps({"done":True,"response":'{"SUPPORT":0.6,"CONTRADICT":0.1,"IRRELEVANT":0.3}',
                                    "thinking":"REASONING_SECRET"}).encode())
    monkeypatch.setattr(ollama_backend,"urlopen",transport)
    first=stance_smoke.run_smoke(root=tmp_path,inputs_path=ipath,output_dir=tmp_path/"run1",cache_dir=tmp_path/"cache")
    assert first["summary"]["model_calls"]==6 and first["summary"]["successes"]==6
    assert first["summary"]["pubmed_requests"]==0
    for body,row in zip(captured,rows):
        assert body["think"] is True and body["options"]=={"temperature":0,"seed":42}
        prompt=body["prompt"]
        payload=json.loads(prompt.removeprefix(stance_smoke.STANCE_PROMPT))
        assert payload=={k:row[k] for k in stance_smoke.MODEL_INPUT_FIELDS}
        assert payload["candidate_option_text"]==row["candidate_option_text"]
        assert set(payload)==set(stance_smoke.MODEL_INPUT_FIELDS)
    second=stance_smoke.run_smoke(root=tmp_path,inputs_path=ipath,output_dir=tmp_path/"run2",cache_dir=tmp_path/"cache")
    assert second["summary"]["model_calls"]==0 and second["summary"]["cache_hits"]==6
    assert len(captured)==6
    assert all("REASONING_SECRET" not in p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    with pytest.raises(ValueError):
        stance_smoke.run_smoke(root=tmp_path,inputs_path=ipath,output_dir=tmp_path/"run1",cache_dir=tmp_path/"cache")


def test_guard_rejects_additional_candidate_before_transport(monkeypatch):
    backend=ollama_backend.OllamaTextGenerationBackend(think=True)
    sent=[]
    monkeypatch.setattr(ollama_backend,"urlopen",lambda *args,**kwargs:sent.append(args))
    row=inputs()[0]
    with stance_smoke._isolated_requests(backend,row,[]):
        with pytest.raises(ValueError):
            backend.generate(stance_smoke.STANCE_PROMPT+json.dumps({**{k:row[k] for k in stance_smoke.MODEL_INPUT_FIELDS},"other_options":["secret"]}))
    assert not sent

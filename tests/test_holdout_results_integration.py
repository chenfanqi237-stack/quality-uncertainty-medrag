"""Synthetic CPU-only tests; no model, private records or research annotations."""
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location("integration", Path(__file__).resolve().parents[1]/"cloud/integrate_holdout_results_v1.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def zip_bytes(files, wrong=False, extra=False):
    stream=io.BytesIO()
    manifest={"files":{k:m.digest(v) for k,v in files.items()}}
    if wrong:
        manifest["files"][next(iter(files))]="0"*64
    with zipfile.ZipFile(stream,"w") as z:
        for k,v in files.items():
            z.writestr(k,v)
        z.writestr("manifest.json",json.dumps(manifest))
        if extra:
            z.writestr("unexpected",b"payload")
    return stream.getvalue()


def predictions(answer=None):
    result={}
    for q in m.QIDS:
        method={"answer":{"status":"ANSWERED" if answer else "ABSTAIN","selected_answer":answer,
                           "reason":"POSITIVE_SUPPORT" if answer else "EMPTY_EVIDENCE_POOL","rankings":[]},
                "option_scores":{o:{"score":None,"native_claim_decision":"ABSTAIN","reason":"EMPTY_EVIDENCE_POOL"} for o in "ABCDE"}}
        result[q]={"status":"COMPLETE","conditions":{c:{"methods":{a:json.loads(json.dumps(method)) for a in "ABCD"}} for c in m.CONDITIONS}}
    return result


@pytest.mark.parametrize("names",[["../a"],["/a"],["a\\b"],["C:/a"],["a","a"],["a","A"]])
def test_reject_unsafe_archive_names(names):
    with pytest.raises(ValueError):
        m.safe_names(names)


def test_member_hash_and_crc_valid():
    with zipfile.ZipFile(io.BytesIO(zip_bytes({"data":b"fixture"}))) as z:
        assert m.verify_zip(z,"manifest.json")["files"]["data"] == m.digest(b"fixture")


@pytest.mark.parametrize("kwargs",[{"wrong":True},{"extra":True}])
def test_corrupt_or_unmanifested_member_rejected(kwargs):
    with zipfile.ZipFile(io.BytesIO(zip_bytes({"data":b"fixture"},**kwargs))) as z:
        with pytest.raises(ValueError):
            m.verify_zip(z,"manifest.json")


def test_exact_holdout_scope():
    assert len(m.QIDS)==20 and len(set(m.QIDS))==20
    assert tuple(int(q.rsplit("-",1)[1]) for q in m.QIDS)==tuple(range(31,51))


def test_gold_gate_precedes_any_read():
    with pytest.raises(ValueError,match="Gold forbidden"):
        m.read_gold_after_reproof(Path("nonexistent"),None,[],False)


def test_incomplete_is_not_abstention():
    p=predictions(); p[m.QIDS[0]]["status"]="INCOMPLETE"
    with pytest.raises(ValueError,match="Incomplete"):
        m.method_metrics(p,{q:"A" for q in m.QIDS},"B","quality_disabled")


def test_missing_question_blocks_evaluation():
    p=predictions();p.pop(m.QIDS[0])
    with pytest.raises(ValueError,match="scope"):
        m.method_metrics(p,{q:"A" for q in m.QIDS},"B","quality_disabled")


def test_no_answers_metrics_undefined_not_zero():
    result=m.method_metrics(predictions(),{q:"A" for q in m.QIDS},"C","quality_disabled")
    assert result["answered_accuracy"] is None and result["selective_risk"] is None
    assert result["answered_accuracy_ci95"] is None and result["undefined_scores"]==100


def test_primary_and_supplementary_intervals():
    r=m.paired_intervals(4,3,0,13,5,1)
    assert r["coverage"]["interval95"] == pytest.approx([-0.20661428711733376,0.5219944465456091],abs=1e-12)
    assert r["accuracy"]["interval95"] == pytest.approx([-0.03714213705333805,0.33130249931992706],abs=1e-12)
    assert r["coverage"]["difference"]==.2 and r["accuracy"]["difference"]==.15


@pytest.mark.parametrize("k,n,expected",[(0,20,[0.,.16843347098308534]),(20,20,[.8315665290169146,1.]),(1,1,[.025,1.])])
def test_cp_edges(k,n,expected):
    assert m.cp_interval(k,n)==pytest.approx(expected,abs=1e-12)


def test_csv_roundtrip_blanks_precision(tmp_path):
    rows=[{"id":"case","value":7/13,"flag":None}]
    path=tmp_path/"result.csv";m.export_csv(path,rows)
    assert "0.5384615384615384" in path.read_text() and path.read_text().endswith(",\n")


def test_source_files_never_overwritten(tmp_path):
    path=tmp_path/"immutable";m.new_file(path,b"source")
    with pytest.raises(FileExistsError):
        m.new_file(path,b"replacement")
    assert path.read_bytes()==b"source"


def test_numeric_disagreement_blocks():
    with pytest.raises(ValueError,match="disagreement"):
        m.close_numbers([.4],[.35])


def test_transition_abstention_not_wrong_answer():
    p=predictions();q=m.QIDS[0]
    p[q]["conditions"]["quality_disabled"]["methods"]["C"]["answer"].update(status="ANSWERED",selected_answer="A")
    rows=m.transitions(p,{x:"A" for x in m.QIDS},"B","C","quality_disabled")
    assert rows[0]["transition"]=="abstain_to_answer" and rows[0]["source_correct_if_answered"] is None
    assert rows[0]["destination_correct_if_answered"] is True


def test_vanilla_excluded_from_holdout_metrics():
    assert all(method in "ABCD" for method,_ in m.METHODS) and len(m.METHODS)==7


def test_score_tolerance_and_ranking_semantics():
    p=predictions();q=m.QIDS[0]
    for a in "CD":
        p[q]["conditions"]["quality_disabled"]["methods"][a]["option_scores"]["A"].update(score=.1)
    p[q]["conditions"]["quality_disabled"]["methods"]["D"]["option_scores"]["A"]["score"] += 1e-13
    assert m.contrast(p,"C","D","quality_disabled")["score_changes"]==0

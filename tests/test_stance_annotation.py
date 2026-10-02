import csv
import json
from pathlib import Path

import pytest

from quality_uncertainty_medrag import stance_annotation as annotation


class GoldForbidden(dict):
    def __getitem__(self,key):
        assert key in {"id","question","options"}, "Gold/correctness/metadata access is forbidden"
        return super().__getitem__(key)


def question(index):
    return {"id":f"medqa-us-dev-{index:06d}","question":f"Clinical question {index}",
        "options":{name:"Candidate "+name for name in "ABCDE"},"answer":"FORBIDDEN_GOLD",
        "answer_idx":"FORBIDDEN_GOLD","metadata":{"upstream":{"answer":"FORBIDDEN_GOLD"}}}


def test_question_projection_never_accesses_gold_or_metadata():
    projected=annotation.project_question(GoldForbidden(question(1)))
    assert set(projected)=={"question_id","question_text","options"}
    assert "FORBIDDEN_GOLD" not in json.dumps(projected)


@pytest.fixture
def sources(tmp_path):
    root=tmp_path/"project"
    path=root/"data/processed/medqa_us_dev_50.jsonl"
    path.parent.mkdir(parents=True)
    # An invalid line 31 proves that it is not read or parsed.
    path.write_text("".join(json.dumps(question(index))+"\n" for index in range(1,31))+"DO_NOT_READ_LINE_31\n",encoding="utf-8")
    for source,start,end in [(annotation.EVIDENCE_FILES[0],1,10),(annotation.EVIDENCE_FILES[1],11,30)]:
        path=root/source;path.parent.mkdir(parents=True,exist_ok=True)
        rows=[]
        for index in range(start,end+1):
            for rank in [1,8,15]:
                rows.append({"question_id":f"medqa-us-dev-{index:06d}","doc_id":f"{index}{rank:02d}","rank":rank,
                    "metadata":{"title":f"Study {index} rank {rank}","abstract":"Frozen abstract.",
                                "quality":"FORBIDDEN_QUALITY","stance":"FORBIDDEN_PREDICTION"},
                    "annotated_stances":"FORBIDDEN_STANCE"})
        path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    return root


def test_question_reader_stops_before_record_31(sources):
    rows,digest=annotation.read_development_questions(sources/"data/processed/medqa_us_dev_50.jsonl")
    assert len(rows)==30 and len(digest)==64
    assert max(rows)=="medqa-us-dev-000030"


def test_sampling_is_deterministic_diverse_unique_and_model_independent(sources):
    questions,_=annotation.read_development_questions(sources/"data/processed/medqa_us_dev_50.jsonl")
    evidence=annotation.read_frozen_evidence([sources/path for path in annotation.EVIDENCE_FILES])
    first,manifest=annotation.sample_blind_pairs(questions,evidence)
    second,manifest2=annotation.sample_blind_pairs(questions,evidence)
    assert first==second and manifest==manifest2
    assert len(first)==30 and len({(r["question_id"],r["candidate_option_id"],r["pmid"]) for r in first})==30
    assert len({r["question_id"] for r in first})==10
    for qid in manifest["question_ids"]:
        assert len({r["candidate_option_id"] for r in first if r["question_id"]==qid})==3
    assert {r["retrieval_rank"] for r in manifest["selected_provenance"]}=={1,8,15}
    assert manifest["used_model_outputs"] is False and manifest["used_gold_or_correctness"] is False


def test_blind_csv_has_exact_schema_blank_labels_and_no_predictions(sources):
    rows,manifest,output=annotation.prepare_annotations(sources)
    with (output/"blind_30.csv").open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        saved=list(reader)
        assert tuple(reader.fieldnames)==annotation.FIELDS
    assert len(saved)==30 and all(row["human_stance"]=="" for row in saved)
    assert "FORBIDDEN" not in (output/"blind_30.csv").read_text()
    assert manifest["human_stance_blank_count"]==30
    guide=(output/"ANNOTATION_GUIDE.md").read_text()
    assert "Evidence supporting a different diagnosis does NOT automatically" in guide
    assert "The evidence does not provide sufficient information to support or" in guide


def test_existing_annotation_and_human_labels_are_never_overwritten(sources):
    _,_,output=annotation.prepare_annotations(sources)
    path=output/"blind_30.csv"
    path.write_text("An existing human label",encoding="utf-8")
    before=path.read_bytes()
    with pytest.raises(ValueError,match="human labels"):
        annotation.prepare_annotations(sources)
    assert path.read_bytes()==before


@pytest.mark.parametrize("qid",["medqa-us-dev-000031","medqa-us-dev-000050","medqa-us-dev-000000"])
def test_non_development_ids_rejected(qid):
    with pytest.raises(ValueError):
        annotation.development_id(qid)

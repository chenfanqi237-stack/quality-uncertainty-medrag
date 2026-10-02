import csv
import json

import pytest

from quality_uncertainty_medrag import stance_annotation as first
from quality_uncertainty_medrag import stance_directional_annotation as directional


def test_informative_overlap_removes_narrative_terms_and_normalizes_plurals():
    assert directional.informative_tokens("The patient study: receptors and antibodies") == {"receptor","antibody"}
    assert not directional.informative_tokens("patient study treatment disease")


def test_relation_cue_requires_candidate_overlap_in_same_text_unit():
    candidate="Sertraline"
    evidence={"rank":1,"evidence_title":"Sertraline dosing.","evidence_abstract":"Another drug was ineffective."}
    separate=directional.features(candidate,evidence,{"sertraline":3})
    assert separate["relation_terms_anywhere_in_evidence"]==["ineffective"]
    assert separate["relation_terms_in_candidate_overlap_context"]==[]
    evidence["evidence_abstract"]="SERTRALINE was ineffective."
    together=directional.features(candidate,evidence,{"sertraline":3})
    assert together["relation_terms_in_candidate_overlap_context"]==["ineffective"]
    assert together["candidate_lexical_overlap_weight"]==3
    assert together["candidate_coverage"]==1


def test_relation_language_remains_a_feature_and_never_a_stance_label():
    found=directional.relation_terms("It is not effective and is contraindicated. It causes harm, decreases risk, and prevents infection.")
    assert set(found)=={"not","contraindicated","causes","decreases","prevents"}
    assert not {"SUPPORT","CONTRADICT","IRRELEVANT"}.intersection(found)


@pytest.fixture
def sources(tmp_path):
    root=tmp_path/"project"
    options={"A":"Cytokine receptor inhibition","B":"Cytokine secretion","C":"Immune modulation",
             "D":"Cytokine binding","E":"Receptor blockade"}
    question_path=root/"data/processed/medqa_us_dev_50.jsonl"
    question_path.parent.mkdir(parents=True)
    raw=[{"id":f"medqa-us-dev-{i:06d}","question":f"Clinical vignette {i}","options":options,
          "answer":"DO_NOT_USE_GOLD","answer_idx":"DO_NOT_USE_GOLD","metadata":{"upstream":{"answer":"DO_NOT_USE_GOLD"}}}
         for i in range(1,31)]
    question_path.write_text("".join(json.dumps(row)+"\n" for row in raw)+"INVALID_LINE_31_DO_NOT_READ\n",encoding="utf-8")
    for rel,low,high in [(first.EVIDENCE_FILES[0],1,10),(first.EVIDENCE_FILES[1],11,30)]:
        path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
        docs=[]
        for i in range(low,high+1):
            for rank in (1,8,15):
                docs.append({"question_id":f"medqa-us-dev-{i:06d}","doc_id":f"{i}{rank:02d}","rank":rank,
                    "metadata":{"title":"Cytokine receptor blockade","abstract":"Receptor inhibition decreases cytokine secretion. Immune modulation is associated with receptor binding.",
                                "quality_score":"DO_NOT_USE_QUALITY","stance":"DO_NOT_USE_PREDICTION"}})
        path.write_text("".join(json.dumps(d)+"\n" for d in docs),encoding="utf-8")
    output=root/"outputs/stance_annotation/dev_1_30";output.mkdir(parents=True)
    old=[]
    for i in range(1,11):
        for rank in (1,8,15):
            old.append({"pair_id":str(len(old)+1),"question_id":f"medqa-us-dev-{i:06d}","question_text":"Previous text",
                "candidate_option_id":"A","candidate_option_text":"Previous candidate","pmid":f"{i}{rank:02d}",
                "evidence_title":"Previous title","evidence_abstract":"Previous abstract","human_stance":"DO_NOT_READ_LABEL"})
    with (output/"blind_30.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=first.FIELDS);writer.writeheader();writer.writerows(old)
    return root


def read_sources(root):
    questions,_=first.read_development_questions(root/"data/processed/medqa_us_dev_50.jsonl")
    evidence=first.read_frozen_evidence([root/rel for rel in first.EVIDENCE_FILES])
    excluded=directional.existing_keys(root/"outputs/stance_annotation/dev_1_30/blind_30.csv")
    return questions,evidence,excluded


def test_fixed_sampling_is_deterministic_order_independent_and_diverse(sources):
    questions,evidence,excluded=read_sources(sources)
    rows,manifest=directional.sample_directional_pairs(questions,evidence,excluded)
    shuffled_questions=dict(reversed(list(questions.items())))
    shuffled_evidence={key:list(reversed(value)) for key,value in reversed(list(evidence.items()))}
    rows2,manifest2=directional.sample_directional_pairs(shuffled_questions,shuffled_evidence,excluded)
    assert rows==rows2 and manifest==manifest2
    assert len(rows)==30
    keys={(r["question_id"],r["candidate_option_id"],r["pmid"]) for r in rows}
    assert len(keys)==30 and not keys.intersection(excluded)
    assert manifest["question_count"]>=10
    assert len({r["candidate_option_id"] for r in rows})>=3
    assert max(manifest["option_label_counts"].values())<=10
    assert all(p["shared_candidate_terms"] for p in manifest["selected_provenance"])
    assert manifest["selected_pairs_with_relation_cooccurrence"]==30
    assert all(sum(r["question_id"]==qid for r in rows)<=3 for qid in manifest["questions_used"])


def test_generated_csv_is_blind_additional_blank_and_preserves_first_set(sources):
    first_path=sources/"outputs/stance_annotation/dev_1_30/blind_30.csv"
    before=first_path.read_bytes()
    rows,manifest,output=directional.prepare_annotations(sources)
    with (output/"blind_directional_30.csv").open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f);saved=list(reader)
        assert tuple(reader.fieldnames)==first.FIELDS
    assert len(saved)==30 and all(r["human_stance"]=="" for r in saved)
    assert "DO_NOT" not in (output/"blind_directional_30.csv").read_text()
    assert first_path.read_bytes()==before
    assert manifest["used_gold_answers"] is False and manifest["used_model_predictions"] is False
    assert manifest["dev_31_50_used"] is False and manifest["question_records_read"]==30
    assert manifest["automatic_stance_assignment"] is False and manifest["model_performance_calculated"] is False
    assert all(set(r)==set(first.FIELDS) for r in rows)


def test_gold_fields_and_prediction_files_cannot_change_sample(sources):
    before=directional.sample_directional_pairs(*read_sources(sources))[0]
    path=sources/"data/processed/medqa_us_dev_50.jsonl"
    raw=[json.loads(line) for line in path.read_text().splitlines()[:30]]
    for row in raw:
        row.update(answer="CHANGED_GOLD",answer_idx="CHANGED_GOLD",metadata={"upstream":{"answer":"CHANGED_GOLD"}})
    path.write_text("".join(json.dumps(row)+"\n" for row in raw)+"INVALID_LINE_31\n",encoding="utf-8")
    predictions=sources/"outputs/stance_nli_probe/unread.json";predictions.parent.mkdir(parents=True)
    predictions.write_text("INVALID_MODEL_OUTPUT_MUST_NOT_BE_READ",encoding="utf-8")
    assert directional.sample_directional_pairs(*read_sources(sources))[0]==before


def test_existing_second_annotations_are_not_overwritten(sources):
    _,_,output=directional.prepare_annotations(sources)
    path=output/"blind_directional_30.csv"
    before=path.read_bytes()
    with pytest.raises(ValueError,match="human labels"):
        directional.prepare_annotations(sources)
    assert path.read_bytes()==before


def test_insufficient_eligible_pool_fails_without_silent_rule_relaxation(sources):
    questions,evidence,excluded=read_sources(sources)
    for docs in evidence.values():
        for doc in docs:
            doc.update(evidence_title="Unrelated paper",evidence_abstract="Unrelated abstract.")
    with pytest.raises(ValueError,match="no labels, manual repairs, or silent relaxation"):
        directional.sample_directional_pairs(questions,evidence,excluded)


def test_out_of_range_question_is_rejected_before_sampling(sources):
    questions,evidence,excluded=read_sources(sources)
    questions["medqa-us-dev-000031"]={"question_text":"Synthetic forbidden range","options":{"A":"Cytokine"}}
    with pytest.raises(ValueError,match="1-30"):
        directional.sample_directional_pairs(questions,evidence,excluded)

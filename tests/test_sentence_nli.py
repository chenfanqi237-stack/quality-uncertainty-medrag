import copy
import math
from types import SimpleNamespace

import pytest

from quality_uncertainty_medrag import nli_stance as nli
from quality_uncertainty_medrag import nli_truncation_control as control
from quality_uncertainty_medrag import sentence_nli as sentence
from quality_uncertainty_medrag import stance_sentence_probe as probe
from test_nli_truncation_control import base_backend, Tokenizer, previous_run

INPUTS=dict(question_stem="What intervention is indicated?",candidate_option_text="Only this candidate",
            evidence_title="Title. Still one title.",evidence_abstract="First finding. Second finding.")
IDS=dict(question_id="medqa-us-dev-000004",candidate_option_id="A",evidence_doc_id="123")


class SingleTokenizer(Tokenizer):
    def __call__(self,premise,hypothesis="",**kwargs):
        return super().__call__(premise,hypothesis,**kwargs)


def backend():
    base=base_backend()
    base.tokenizer=SingleTokenizer()
    return control.SupportedContextNLIBackend(base)


@pytest.mark.parametrize("text,expected",[
    ("",[]),("  Unpunctuated tail  ",["Unpunctuated tail"]),
    ("First! Second? Third.",["First!","Second?","Third."]),
    ("Dr. Smith measured 3.5 mg. Results improved.",["Dr. Smith measured 3.5 mg.","Results improved."]),
    ("Examples, e.g. fever, occurred. A. Brown agreed.",["Examples, e.g. fever, occurred.","A. Brown agreed."]),
    ('He wrote "No benefit." Next result.', ['He wrote "No benefit."','Next result.']),
    ("First finding.\n\nSecond  finding.",["First finding.","Second  finding."]),
])
def test_sentence_splitting_is_deterministic_and_preserves_text(text,expected):
    assert sentence.split_abstract(text)==sentence.split_abstract(text)==expected


def test_title_is_one_candidate_and_abstract_order_is_preserved():
    rows=sentence.evidence_sentences(title=INPUTS["evidence_title"],abstract=INPUTS["evidence_abstract"])
    assert [r["sentence_index"] for r in rows]==[0,1,2]
    assert [r["source"] for r in rows]==["title","abstract","abstract"]
    assert [r["text"] for r in rows]==[INPUTS["evidence_title"],"First finding.","Second finding."]
    assert len(sentence.evidence_sentences(title="Title",abstract=""))==1


@pytest.mark.parametrize("scores,strength",[(dict(SUPPORT=.2,CONTRADICT=.7,IRRELEVANT=.1),.7),
    (dict(SUPPORT=.45,CONTRADICT=.05,IRRELEVANT=.5),.45)])
def test_directional_strength_is_maximum_of_two_directional_scores(scores,strength):
    assert sentence.directional_strength(scores)==strength


def judged(index,support,contradict,irrelevant):
    logits=[math.log(support),math.log(irrelevant),math.log(contradict)]
    scores=nli.softmax_logits(logits)
    return {"status":"SUCCESS","sentence_index":index,**nli.score_result(logits,scores,backend().label_mapping)}


def test_selected_distribution_keeps_irrelevant_argmax():
    rows=[judged(0,.45,.05,.5),judged(1,.4,.1,.5)]
    selected=sentence.select_sentence(rows)
    assert selected is rows[0]
    assert selected["argmax_label"]=="IRRELEVANT"
    assert selected["nli_label_scores"]["IRRELEVANT"]==pytest.approx(.5)


def test_highest_directional_strength_selects_contradict_and_exact_ties_use_first():
    rows=[judged(0,.2,.1,.7),judged(1,.1,.8,.1),judged(2,.1,.8,.1)]
    assert sentence.select_sentence(rows) is rows[1]
    assert sentence.select_sentence(rows)["argmax_label"]=="CONTRADICT"


def test_partial_sentence_failure_does_not_select_from_incomplete_set():
    with pytest.raises(nli.NLIProbeError):
        sentence.select_sentence([judged(0,.6,.1,.3),{"status":"MODEL_FAILURE","sentence_index":1}])


def test_cache_key_includes_versions_effective_length_and_original_source_text():
    wrapped=sentence.SentenceBackend(backend())
    sentences=sentence.evidence_sentences(title=INPUTS["evidence_title"],abstract=INPUTS["evidence_abstract"])
    context=sentence.cache_context(ids=IDS,inputs=INPUTS,sentences=sentences,backend_identity=wrapped.identity)
    key=nli.text_hash(nli.canonical(context))
    assert context["effective_max_length"]==8192
    for field in ["model_revision","sentence_split_version","sentence_selection_version"]:
        identity={**wrapped.identity,field:"different"}
        changed=sentence.cache_context(ids=IDS,inputs=INPUTS,sentences=sentences,backend_identity=identity)
        assert nli.text_hash(nli.canonical(changed))!=key
    identity=copy.deepcopy(wrapped.identity);identity["tokenizer_settings"]["max_length"]=4096
    assert nli.text_hash(nli.canonical(sentence.cache_context(ids=IDS,inputs=INPUTS,sentences=sentences,backend_identity=identity)))!=key
    with pytest.raises(ValueError):
        sentence.cache_context(ids=IDS,inputs={**INPUTS,"answer":"secret"},sentences=sentences,backend_identity=wrapped.identity)


def test_every_sentence_is_cached_and_receives_same_hypothesis_only(tmp_path):
    real_backend=backend()
    classifier=sentence.SentenceNLIClassifier(real_backend,cache_dir=tmp_path/sentence.CACHE_NAMESPACE)
    first=classifier.classify_texts(**IDS,**INPUTS)
    assert first["sentence_count"]==3 and first["model_calls"]==3
    expected=nli.build_hypothesis(question_stem=INPUTS["question_stem"],candidate_option_text=INPUTS["candidate_option_text"])
    assert all(r["pair"]["hypothesis"]==expected for r in first["sentences"])
    assert [r["pair"]["premise"] for r in first["sentences"]]==[r["text"] for r in first["sentences"]]
    assert all(r["truncation"]["effective_max_length"]==8192 for r in first["sentences"])
    assert first["nli_label_scores"]==first["sentences"][0]["nli_label_scores"]
    before={p:p.read_bytes() for p in (tmp_path/sentence.CACHE_NAMESPACE).rglob("*.json")}
    cached=classifier.classify_texts(**IDS,**INPUTS)
    assert cached["cache_status"]=="HIT" and cached["model_calls"]==0
    assert len(real_backend.model.calls)==3
    assert {p:p.read_bytes() for p in before}==before


def test_full_classifier_preserves_selected_non_one_hot_irrelevant_distribution(tmp_path):
    real_backend=backend()
    def infer(*,premise,hypothesis):
        support,contradict,irrelevant=(.45,.05,.5) if premise==INPUTS["evidence_title"] else (.4,.1,.5)
        logits=[math.log(support),math.log(irrelevant),math.log(contradict)]
        return {**nli.score_result(logits,nli.softmax_logits(logits),real_backend.label_mapping),
                "truncation":{"effective_max_length":8192,"truncation_occurred":False}}
    real_backend.infer=infer
    classifier=sentence.SentenceNLIClassifier(real_backend,cache_dir=tmp_path/sentence.CACHE_NAMESPACE)
    result=classifier.classify_texts(**IDS,**INPUTS)
    assert result["selected_source"]=="title"
    assert result["argmax_label"]=="IRRELEVANT"
    assert result["nli_label_scores"]==pytest.approx(dict(SUPPORT=.45,CONTRADICT=.05,IRRELEVANT=.5))


@pytest.mark.parametrize("field",["answer","answer_idx","options","metadata","candidate_correctness"])
def test_no_gold_or_other_options_can_enter_classifier(tmp_path,field):
    real_backend=backend()
    classifier=sentence.SentenceNLIClassifier(real_backend,cache_dir=tmp_path/sentence.CACHE_NAMESPACE)
    with pytest.raises(TypeError):
        classifier.classify_texts(**IDS,**INPUTS,**{field:"SECRET_GOLD_OR_OPTIONS"})
    assert real_backend.model.calls==[]


def test_diagnostic_uses_exact_eighteen_controls_and_preserves_previous_outputs(previous_run):
    root,source,inputs,previous,base=previous_run
    base.tokenizer=SingleTokenizer()
    full_dir=root/probe.FULL_CONTEXT_RUN
    control.run_control(root=root,output_dir=full_dir,backend_loader=lambda **kwargs:base)
    base.model.calls.clear()
    before={p:p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report,output=probe.run_probe(root=root,backend_loader=lambda **kwargs:base)
    assert report["status"]=="COMPLETE"
    assert report["summary"]["judgments"]==18
    assert tuple(original_key(r) for r in report["results"])==probe.original.DIAGNOSTIC_KEYS
    assert all(r["hypothesis"]==r["previous_256"]["pair"]["hypothesis"] for r in report["results"])
    assert all(r["previous_8192"]["pair"]==r["previous_256"]["pair"] for r in report["results"])
    assert all(p.read_bytes()==data for p,data in before.items())
    assert "selected sentence" in (output/"comparison.md").read_text()
    assert report["summary"]["accuracy_evaluated"] is False


def original_key(row):
    return tuple(row[name] for name in nli.ID_FIELDS)

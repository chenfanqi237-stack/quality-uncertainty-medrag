"""Text-only enrichment of a second blinded development annotation sample."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from . import stance_annotation as annotation

VERSION = "blind-directional-dev-30-lexical-structural-v1"
STOPWORDS = frozenset("""
a an the of to in on at by for from with without into and or but as is are was were
be been being do does did can could may might will would should must have has had
it its this that these those their them they he she his her we our you your which
what when where how who why than then there here only most more less many some any
all each other another same such also both either neither one two three four five
first second third following about between after before during through within
patient patients man men woman women child children year years old age aged
clinical medical physician doctor diagnosis diagnosed disease diseases syndrome
condition conditions symptom symptoms treatment therapy management study studies
result results finding findings evidence group groups report reported reports
case cases level levels normal abnormal high low increased decreased primary
secondary specific general significant significantly effect effects compared
comparison associated association mechanism mechanisms due caused causes cause
increase increases decrease decreases prevent prevents prevention not no effective
ineffective contraindicated versus using used use include includes including
usually commonly several can likely best appropriate indicated recommended
""".split())
RELATIONS = {
    "not": r"\bnot\b", "ineffective": r"\bineffective\b", "contraindicated": r"\bcontraindicated\b",
    "associated with": r"\bassociated\s+with\b",
    "increases": r"\bincreas(?:e|es|ed|ing)\b", "decreases": r"\bdecreas(?:e|es|ed|ing)\b",
    "prevents": r"\bprevent(?:s|ed|ing)?\b", "causes": r"\bcaus(?:e|es|ed|ing)\b",
    "treatment of": r"\btreatment\s+of\b", "management of": r"\bmanagement\s+of\b",
    "mechanism": r"\bmechanisms?\b",
}
MAX_PER_QUESTION = 3
MAX_PER_QUESTION_OPTION = 2
MAX_PER_QUESTION_ARTICLE = 2
MAX_PER_OPTION_LABEL = 10


def informative_tokens(text):
    annotation._string(text, "heuristic text", empty=True)
    tokens = set()
    for token in re.findall(r"[a-z]+", text.lower()):
        if token in STOPWORDS or len(token) < 3:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "is", "us")):
            token = token[:-1]
        if token not in STOPWORDS:
            tokens.add(token)
    return tokens


def relation_terms(text):
    return sorted(name for name, pattern in RELATIONS.items() if re.search(pattern, text, flags=re.IGNORECASE))


def features(candidate, evidence, idf):
    candidate_terms = informative_tokens(candidate)
    title_terms = informative_tokens(evidence["evidence_title"])
    text = evidence["evidence_title"] + "\n" + evidence["evidence_abstract"]
    common = candidate_terms & informative_tokens(text)
    contextual = set()
    # Fixed punctuation/newline context units, not an NLI sentence splitter.
    for unit in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
        if common & informative_tokens(unit):
            contextual.update(relation_terms(unit))
    return {
        "shared_candidate_terms": sorted(common),
        "candidate_term_count": len(candidate_terms),
        "candidate_coverage": len(common) / len(candidate_terms) if candidate_terms else 0.0,
        "candidate_lexical_overlap_weight": math.fsum(idf.get(term, 1.0) for term in sorted(common)),
        "shared_title_terms": sorted(candidate_terms & title_terms),
        "relation_terms_in_candidate_overlap_context": sorted(contextual),
        "relation_terms_anywhere_in_evidence": relation_terms(text),
        "retrieval_rank": evidence["rank"],
    }


def existing_keys(csv_path):
    keys = set()
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"question_id", "candidate_option_id", "pmid"}.issubset(reader.fieldnames or []):
            raise ValueError("First annotation CSV must identify its judgment triples")
        for row in reader:
            # Human labels, text, and any other columns are not accessed.
            qid = annotation.development_id(row["question_id"])
            keys.add((qid, annotation._string(row["candidate_option_id"], "option ID"),
                      annotation._string(row["pmid"], "PMID")))
    return keys


def sample_directional_pairs(questions, evidence, excluded, seed=42):
    frequencies = Counter()
    document_count = 0
    for qid in sorted(evidence):
        annotation.development_id(qid)
        for article in sorted(evidence[qid], key=lambda row:(row["rank"], row["pmid"])):
            frequencies.update(informative_tokens(article["evidence_title"] + "\n" + article["evidence_abstract"]))
            document_count += 1
    idf = {term: math.log((document_count + 1) / (count + 1)) + 1 for term, count in frequencies.items()}
    pool = []
    for qid in sorted(questions):
        annotation.development_id(qid)
        question = questions[qid]
        for label, candidate in sorted(question["options"].items()):
            for article in sorted(evidence.get(qid, []), key=lambda row:(row["rank"], row["pmid"])):
                key = (qid, label, article["pmid"])
                if key in excluded:
                    continue
                info = features(candidate, article, idf)
                if not info["shared_candidate_terms"]:
                    continue
                row = {"question_id":qid, "question_text":question["question_text"],
                       "candidate_option_id":label, "candidate_option_text":candidate,
                       "pmid":article["pmid"], "evidence_title":article["evidence_title"],
                       "evidence_abstract":article["evidence_abstract"], "human_stance":""}
                pool.append({"key":key, "row":row, "features":info})
    if len({item["key"] for item in pool}) != len(pool):
        raise ValueError("Frozen candidate pool contains duplicate triples")
    rng = random.Random(seed)
    # Canonical traversal makes tied ranking invariant to source container order.
    for item in sorted(pool, key=lambda item:item["key"]):
        item["tie_order"] = rng.random()
    def priority(item):
        info = item["features"]
        return (-bool(info["relation_terms_in_candidate_overlap_context"]),
                -info["candidate_lexical_overlap_weight"], -info["candidate_coverage"],
                -len(info["shared_title_terms"]), info["retrieval_rank"], item["tie_order"], item["key"])
    question_counts, option_counts, article_counts, label_counts = Counter(), Counter(), Counter(), Counter()
    chosen = []
    for item in sorted(pool, key=priority):
        qid, label, pmid = item["key"]
        if question_counts[qid] >= MAX_PER_QUESTION or option_counts[qid,label] >= MAX_PER_QUESTION_OPTION or article_counts[qid,pmid] >= MAX_PER_QUESTION_ARTICLE or label_counts[label] >= MAX_PER_OPTION_LABEL:
            continue
        chosen.append(item)
        question_counts[qid] += 1; option_counts[qid,label] += 1; article_counts[qid,pmid] += 1; label_counts[label] += 1
        if len(chosen) == 30:
            break
    if len(chosen) != 30:
        raise ValueError("Fixed text-only eligibility and diversity rules cannot yield 30 additional pairs; no labels, manual repairs, or silent relaxation were used")
    rng.shuffle(chosen)
    rows = [{"pair_id":f"blind-directional-dev-{index:04d}", **item["row"]} for index,item in enumerate(chosen,1)]
    manifest = {
        "sampling_version":VERSION, "seed":seed, "pair_count":30,
        "sampling_rule":"Require at least one shared informative candidate/evidence token. Rank lexicographically by relation-language co-occurrence, corpus-IDF overlap, candidate coverage, title overlap, retrieval rank, seeded tie order. Apply diversity caps, take 30, then shuffle presentation.",
        "normalization":"Lowercase alphabetic tokens; discard words shorter than 3 and fixed stopwords; -ies -> -y, conservative trailing -s removal except -ss/-is/-us. No synonyms, embeddings, LLM or medical-entity classifier.",
        "stopwords":sorted(STOPWORDS), "relation_patterns":RELATIONS,
        "relation_context_units":"Split title/abstract at terminal punctuation followed by whitespace or at newline; count relation terms only in units containing a shared candidate token.",
        "overlap_weight_formula":"sum(log((N+1)/(document_frequency(term)+1))+1) over unique shared informative terms",
        "idf_corpus_unit":"Distinct frozen (question_id, PMID) records; no stance or quality field used",
        "corpus_evidence_record_count":document_count, "eligible_additional_pair_count":len(pool),
        "diversity_caps":{"per_question":MAX_PER_QUESTION,"per_question_option":MAX_PER_QUESTION_OPTION,
                          "per_question_article":MAX_PER_QUESTION_ARTICLE,"per_option_label_global":MAX_PER_OPTION_LABEL},
        "question_count":len(question_counts), "questions_used":sorted(question_counts),
        "option_label_counts":dict(sorted(label_counts.items())),
        "excluded_first_sample_triple_count":len(excluded),
        "excluded_keys_sha256":hashlib.sha256(json.dumps(sorted(excluded),separators=(",",":")).encode()).hexdigest(),
        "all_selected_pairs_have_candidate_evidence_lexical_overlap":True,
        "selected_pairs_with_relation_cooccurrence":sum(bool(item["features"]["relation_terms_in_candidate_overlap_context"]) for item in chosen),
        "used_gold_answers":False, "used_answer_idx":False, "used_candidate_correctness":False,
        "used_model_predictions":False, "stance_classifier_calls":0, "dev_31_50_used":False,
        "automatic_stance_assignment":False, "model_performance_calculated":False,
        "human_stance_blank_count":30,
        "enrichment_is_a_sampling_intention_not_verified_stance":True,
        "selected_provenance":[{"pair_id":row["pair_id"], "question_id":row["question_id"],
            "candidate_option_id":row["candidate_option_id"], "pmid":row["pmid"], **item["features"]}
            for row,item in zip(rows,chosen)],
    }
    return rows, manifest


def prepare_annotations(root):
    root = Path(root).resolve()
    output = root / "outputs/stance_annotation/dev_1_30"
    csv_path = output / "blind_directional_30.csv"
    manifest_path = output / "directional_sampling_manifest.json"
    if csv_path.exists() or manifest_path.exists():
        raise ValueError("Refusing to overwrite an existing second annotation set or human labels")
    questions, prefix_hash = annotation.read_development_questions(root/"data/processed/medqa_us_dev_50.jsonl")
    source_paths = [root/path for path in annotation.EVIDENCE_FILES]
    evidence = annotation.read_frozen_evidence(source_paths)
    excluded = existing_keys(output/"blind_30.csv")
    rows, manifest = sample_directional_pairs(questions, evidence, excluded)
    manifest.update(question_source="data/processed/medqa_us_dev_50.jsonl", question_records_read=30,
        first_30_question_record_bytes_sha256=prefix_hash,
        evidence_source_sha256={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        first_annotation_source="outputs/stance_annotation/dev_1_30/blind_30.csv",
        first_annotation_fields_used=["question_id","candidate_option_id","pmid"], csv_fields=list(annotation.FIELDS))
    output.mkdir(parents=True,exist_ok=True)
    with csv_path.open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=annotation.FIELDS,extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)
    manifest["csv_sha256"]=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    with manifest_path.open("x",encoding="utf-8",newline="\n") as handle:
        handle.write(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    return rows,manifest,output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    args=parser.parse_args(argv)
    _,manifest,output=prepare_annotations(args.project_root)
    print(json.dumps({name:manifest[name] for name in ("pair_count","question_count","questions_used",
        "eligible_additional_pair_count","selected_pairs_with_relation_cooccurrence","human_stance_blank_count",
        "used_gold_answers","used_model_predictions","dev_31_50_used")},indent=2))
    print("New blinded CSV:",output/"blind_directional_30.csv")
    print("Sampling manifest:",output/"directional_sampling_manifest.json")


if __name__=="__main__":
    main()

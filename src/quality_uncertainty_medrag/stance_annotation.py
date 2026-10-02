"""Output-independent structural sampling of 30 blinded development pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

VERSION = "blind-dev-30-structural-sampling-v1"
FIELDS = ("pair_id", "question_id", "question_text", "candidate_option_id", "candidate_option_text",
          "pmid", "evidence_title", "evidence_abstract", "human_stance")
EVIDENCE_FILES = (
    "data/retrieved/pubmed_medqa_us_dev_10_llm.jsonl",
    "outputs/heldout_retrieval/dev_11_30/500a1f067a9f7826/runs/kaggle-20260928T142621Z/evidence.jsonl",
)
GUIDE = """# Blinded human stance annotation

Read the clinical question, the single candidate option, and the evidence title
and abstract. Enter exactly SUPPORT, CONTRADICT, or IRRELEVANT in human_stance.
The file contains no gold answers or model predictions.

SUPPORT:
The evidence provides information that supports the candidate answer in
the context of the clinical question.

CONTRADICT:
The evidence provides information that conflicts with or argues against
the candidate answer in the context of the clinical question.

IRRELEVANT:
The evidence does not provide sufficient information to support or
contradict the candidate answer.

Important annotation rule:
Evidence supporting a different diagnosis does NOT automatically
contradict the candidate unless the evidence actually provides information
inconsistent with the candidate.
"""


def _string(value, name, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(name + " must be a string" + ("" if empty else " containing text"))
    return value


def development_id(value):
    _string(value, "question_id")
    match = re.fullmatch(r"medqa-us-dev-(\d{6})", value)
    if not match or not 1 <= int(match.group(1)) <= 30:
        raise ValueError("Only already exposed development questions 1-30 are allowed")
    return value


def project_question(raw):
    # Only these keys are accessed; answer, metadata and correctness are unused.
    qid = development_id(raw["id"])
    question = _string(raw["question"], "question")
    options = raw["options"]
    if not isinstance(options, dict) or len(options) < 3:
        raise ValueError("Expected at least three labeled development options")
    projected = { _string(label, "option label"): _string(text, "option text") for label,text in options.items() }
    return {"question_id": qid, "question_text": question, "options": projected}


def read_development_questions(path):
    questions, raw_bytes = {}, bytearray()
    with Path(path).open("rb") as handle:
        # islice reads exactly these 30 records; line 31 is never consumed.
        for line in itertools.islice(handle, 30):
            raw_bytes.extend(line)
            row = project_question(json.loads(line))
            if row["question_id"] in questions:
                raise ValueError("Duplicate development question")
            questions[row["question_id"]] = row
    expected = {f"medqa-us-dev-{index:06d}" for index in range(1,31)}
    if set(questions) != expected:
        raise ValueError("The first 30 records must be exactly development questions 1-30")
    return questions, hashlib.sha256(raw_bytes).hexdigest()


def read_frozen_evidence(paths):
    grouped, keys, ranks = {}, set(), set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                qid = development_id(raw["question_id"])
                pmid = _string(raw["doc_id"], "PMID")
                rank = raw["rank"]
                if type(rank) is not int or rank <= 0:
                    raise ValueError("Evidence rank must be a positive integer")
                if (qid,pmid) in keys or (qid,rank) in ranks:
                    raise ValueError("Duplicate frozen evidence identity or rank")
                metadata = raw["metadata"]
                evidence = {"question_id": qid, "pmid": pmid, "rank": rank,
                    "evidence_title": _string(metadata["title"], "title"),
                    "evidence_abstract": _string(metadata["abstract"], "abstract", empty=True)}
                grouped.setdefault(qid,[]).append(evidence)
                keys.add((qid,pmid)); ranks.add((qid,rank))
    for evidence in grouped.values():
        evidence.sort(key=lambda row:(row["rank"],row["pmid"]))
    return grouped


def sample_blind_pairs(questions, evidence, seed=42):
    """Ten questions x three distinct options and first/middle/last frozen ranks."""
    eligible = sorted(qid for qid in questions if len(evidence.get(qid,[])) >= 3)
    if len(eligible) < 10:
        raise ValueError("At least ten development questions with three frozen articles are required")
    rng = random.Random(seed)
    chosen = rng.sample(eligible,10)
    rows, provenance = [], []
    for qid in chosen:
        question = questions[qid]
        articles = evidence[qid]
        documents = [articles[0], articles[len(articles)//2], articles[-1]]
        labels = rng.sample(list(question["options"]),3)
        for label,document in zip(labels,documents):
            rows.append({"question_id": qid, "question_text": question["question_text"],
                "candidate_option_id": label, "candidate_option_text": question["options"][label],
                "pmid": document["pmid"], "evidence_title": document["evidence_title"],
                "evidence_abstract": document["evidence_abstract"], "human_stance": ""})
            provenance.append({"question_id":qid,"candidate_option_id":label,"pmid":document["pmid"],"retrieval_rank":document["rank"]})
    rng.shuffle(rows)
    rows = [{"pair_id":f"blind-dev-{index:04d}",**row} for index,row in enumerate(rows,1)]
    keys = {(r["question_id"],r["candidate_option_id"],r["pmid"]) for r in rows}
    if len(keys) != 30:
        raise ValueError("Annotation triples must be unique")
    return rows, {"sampling_version":VERSION,"seed":seed,"pair_count":30,
        "question_count":10,"question_ids":sorted(chosen),"sampling_rule":"Seeded structural sample of 10 eligible questions; three distinct options paired with first/middle/last article ranks; shuffled presentation",
        "used_gold_or_correctness":False,"used_model_outputs":False,"dev_31_50_used":False,"selected_provenance":provenance}


def prepare_annotations(root):
    root = Path(root).resolve()
    output = root / "outputs/stance_annotation/dev_1_30"
    destinations = [output/name for name in ("blind_30.csv","ANNOTATION_GUIDE.md","sampling_manifest.json")]
    if any(path.exists() for path in destinations):
        raise ValueError("Refusing to overwrite existing annotation files or human labels")
    questions, prefix_hash = read_development_questions(root/"data/processed/medqa_us_dev_50.jsonl")
    evidence_paths = [root/path for path in EVIDENCE_FILES]
    rows, manifest = sample_blind_pairs(questions,read_frozen_evidence(evidence_paths))
    manifest.update(question_source="data/processed/medqa_us_dev_50.jsonl",question_records_read=30,
        first_30_question_record_bytes_sha256=prefix_hash,
        evidence_source_sha256={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence_paths},
        csv_fields=list(FIELDS),human_stance_blank_count=30,model_evaluation_run=False)
    output.mkdir(parents=True,exist_ok=True)
    with destinations[0].open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS,extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)
    with destinations[1].open("x",encoding="utf-8",newline="\n") as handle:
        handle.write(GUIDE)
    manifest["blind_csv_sha256"]=hashlib.sha256(destinations[0].read_bytes()).hexdigest()
    with destinations[2].open("x",encoding="utf-8",newline="\n") as handle:
        handle.write(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return rows,manifest,output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    args=parser.parse_args(argv)
    _,manifest,output=prepare_annotations(args.project_root)
    print(json.dumps({k:manifest[k] for k in ("pair_count","question_count","question_ids","human_stance_blank_count","used_model_outputs","used_gold_or_correctness","dev_31_50_used")},indent=2))
    print("Annotation output:",output)


if __name__=="__main__":
    main()

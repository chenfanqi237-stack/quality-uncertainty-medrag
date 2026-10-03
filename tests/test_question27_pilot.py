"""Focused CPU-only pilot tests. Synthetic backend responses are NOT research results."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "cloud"))
import question27_pilot as p
import question27_adapter as a


class FakeBackend:
    """In-memory numerical fixture, no service/network/GPU."""
    def __init__(self, raw='{"SUPPORT":1,"CONTRADICT":0,"IRRELEVANT":0}', interrupt=False):
        self.raw, self.interrupt, self.calls = raw, interrupt, 0
        self.requests = []

    def generate(self, prompt, *, generation_config):
        self.calls += 1
        self.requests.append((prompt, generation_config))
        if self.interrupt:
            raise KeyboardInterrupt("synthetic interrupted call")
        return self.raw


def option_values(scores, eligible=True):
    return {o: {"score": s, "eligible": eligible and s is not None and s > 0,
                "native_claim_decision": "SUPPORT" if s is not None and s > 0 else "ABSTAIN"} for o, s in zip(a.OPTIONS, scores)}


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.sampling = p.load_inputs(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="q27-focused-test-")
        self.base = Path(self.temp.name)
        self.state = self.base / "state"
        self.drive = self.base / "drive"
        self.state.mkdir()
        for name in ("runtime", "checkpoints", "final"):
            (self.drive / name).mkdir(parents=True)
        p.sc.write_json(self.state / "pilot_state.json", p.new_meta(self.sampling["pilot_signature"]))
        self.silent = contextlib.redirect_stdout(io.StringIO())
        self.silent.__enter__()

    def tearDown(self):
        self.silent.__exit__(None, None, None)
        self.temp.cleanup()

    def test_scope_and_frozen_quality(self):
        self.assertEqual(len(self.rows), 15)
        self.assertEqual([r["pmid"] for r in self.rows[:3]], ["39709592", "26780003", "21663949"])
        self.assertEqual([r["quality_weight"] for r in self.rows[:3]], [.6666666667, 0, .6666666667])
        self.assertEqual(len(p.expected_inventory(self.rows)), 150)
        self.assertEqual(self.sampling["frozen_sampling_settings"], p.sc.frozen_settings())

    def test_exact_seed_inventory_zero_reuse(self):
        coverage = p.read_json(ROOT / p.PILOT / "sample_coverage.json")
        missing = p.read_json(ROOT / p.PILOT / "missing_samples.json")
        self.assertEqual(coverage["reusable_samples"], 0)
        self.assertEqual(coverage["original_archive_audit"]["all_cache_contexts_validated"], 600)
        self.assertEqual(len(missing), 150)
        self.assertEqual(len({m["cache_key"] for m in missing}), 150)
        for row in self.rows:
            self.assertEqual({i["seed"] for i in missing if i["ids"] == {k: row[k] for k in p.sc.INPUT_FIELDS[:3]}}, set(range(101, 111)))

    def test_runtime_inventory_has_no_gold_or_full_dataset(self):
        m, payload = p.archive_payload(ROOT / "cloud/medqa_question27_k3_runtime_v1.zip", "runtime_manifest.json")
        self.assertFalse(m["medqa_gold_included"])
        self.assertEqual(m["expected_samples"], 150)
        self.assertFalse(any("dev_50.jsonl" in n or "retrieved/" in n or "/reference_60/" in n or n.endswith(".pyc") for n in payload))
        records = [json.loads(line) for line in payload[(p.PILOT / "pilot_input.jsonl").as_posix()].decode().splitlines()]
        self.assertTrue(all(set(r) == p.FIELDS for r in records))
        self.assertEqual({r["question_id"] for r in records}, {p.QID})

    def test_notebook_ten_unexecuted_syntax_valid_cells(self):
        nb = p.read_json(ROOT / "notebooks/colab_medqa_question27_pilot.ipynb")
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 10)
        for c in cells:
            ast.parse("".join(c["source"]))
            self.assertIsNone(c["execution_count"])
            self.assertEqual(c["outputs"], [])
        self.assertIn("prepare_state", "".join(cells[4]["source"]))
        self.assertIn("setup_ollama", "".join(cells[5]["source"]))

    def test_notebook_cells_two_through_five_generation_free_dryrun(self):
        nb = p.read_json(ROOT / "notebooks/colab_medqa_question27_pilot.ipynb")
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        source = "\n".join("".join(c["source"]) for c in cells[1:5])
        temporary_drive = self.base / "notebook-drive"
        (temporary_drive / "runtime").mkdir(parents=True)
        (temporary_drive / "runtime/medqa_question27_k3_runtime_v1.zip").write_bytes((ROOT / "cloud/medqa_question27_k3_runtime_v1.zip").read_bytes())
        source = source.replace("/content/drive/MyDrive/medrag_question27_pilot", temporary_drive.as_posix())
        source = source.replace("/content/medrag_question27_k3_v1", (self.base / "notebook-project").as_posix())
        source = source.replace("/content/medrag_question27_work/state", (self.base / "notebook-state").as_posix())
        source = source.replace("/content/drive/MyDrive/medrag_self_consistency", (self.base / "absent-original").as_posix())
        completed = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", source], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"valid": 0', completed.stdout)
        self.assertIn('"missing": 150', completed.stdout)
        self.assertTrue(list((temporary_drive / "checkpoints").glob("q27_checkpoint_*.zip")))

    def test_strict_frozen_parser_and_modal_tie(self):
        prediction = p.sc.parse_stance_output('{"SUPPORT":0.5,"CONTRADICT":0.5,"IRRELEVANT":0}')
        self.assertIsNone(prediction.label)
        for bad in ('SUPPORT', '{"SUPPORT":1,"CONTRADICT":0,"IRRELEVANT":0,"extra":1}', '{"SUPPORT":NaN,"CONTRADICT":0,"IRRELEVANT":0}'):
            with self.assertRaises(p.sc.StanceOutputError):
                p.sc.parse_stance_output(bad)

    def test_frozen_backend_payload(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"done":true,"response":"{\\"SUPPORT\\":1,\\"CONTRADICT\\":0,\\"IRRELEVANT\\":0}"}'
        with patch("quality_uncertainty_medrag.ollama_backend.urlopen", return_value=Response()) as api:
            backend = p.sc.OllamaTextGenerationBackend(model=p.sc.MODEL, think=True, timeout=600, seed=101)
            raw = backend.generate("synthetic prompt", generation_config={"temperature": .7, "seed": 110})
        payload = json.loads(api.call_args.args[0].data)
        self.assertEqual(payload, {"model": p.sc.MODEL, "prompt": "synthetic prompt", "stream": False, "think": True, "options": {"temperature": .7, "seed": 110}})
        self.assertEqual(p.sc.parse_stance_output(raw).label.value, "SUPPORT")

    def test_unique_positive_answer_and_no_positive_abstention(self):
        result = a.select_answer(option_values([.2, .3, 0, -.1, None]))
        self.assertEqual(result["selected_answer"], "B")
        self.assertEqual(result["undefined_options"], ["E"])
        self.assertEqual(a.select_answer(option_values([0, -.1, None, 0, -.2]))["status"], "ABSTAIN")

    def test_highest_tie_tolerance_and_native_eligibility(self):
        self.assertEqual(a.select_answer(option_values([.3, .3+5e-13, 0, 0, 0]))["reason"], "TIED_HIGHEST_POSITIVE_SCORES")
        self.assertEqual(a.select_answer(option_values([.3, .3+2e-12, 0, 0, 0]))["selected_answer"], "B")
        self.assertEqual(a.select_answer(option_values([.3, 0, 0, 0, 0], eligible=False))["status"], "ABSTAIN")
        self.assertEqual(a.select_answer(option_values([1e-17, 0, 0, 0, 0]))["selected_answer"], "A")

    def test_missing_is_incomplete_not_abstain(self):
        samples = {(p.sc.pair_key(i["row"]), i["seed"]): "IRRELEVANT" for i in p.expected_inventory(self.rows)}
        samples.pop(next(iter(samples)))
        result = a.evaluate(self.rows, samples)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["missing_sample_count"], 1)
        self.assertIsNone(result["predictions"])

    def test_empty_pool_and_zero_quality(self):
        empty = a.evaluate([], {})
        self.assertTrue(all(v["answer"]["reason"] == "EMPTY_EVIDENCE_POOL" for c in empty["conditions"].values() for v in c["methods"].values()))
        rows = [dict(r, quality_weight=0) for r in self.rows[:3]]
        counts = {p.sc.pair_key(r): {"SUPPORT":10,"CONTRADICT":0,"IRRELEVANT":0} for r in rows}
        results, _ = a.aggregate_option(rows, counts, "original_quality")
        self.assertEqual(results["A"]["native_claim_decision"], "SUPPORT")
        self.assertTrue(all(results[m]["score"] is None and not results[m]["eligible"] for m in "BCD"))

    def test_modal_tie_retained_and_same_samples_all_methods(self):
        counts = {p.sc.pair_key(r): {"SUPPORT":5,"CONTRADICT":0,"IRRELEVANT":5} for r in self.rows[:3]}
        results, details = a.aggregate_option(self.rows[:3], counts, "quality_disabled")
        self.assertTrue(all(d["modal_tie"] and d["matched_hard"] is None for d in details))
        self.assertEqual(results["A"]["native_claim_decision"], "ABSTAIN")
        self.assertEqual(results["B"]["native_claim_decision"], "ABSTAIN")
        self.assertEqual(results["C"]["score"], .5)
        self.assertAlmostEqual(results["D"]["score"], .5, delta=1e-12)
        counts[p.sc.pair_key(self.rows[0])] = {"SUPPORT":10,"CONTRADICT":0,"IRRELEVANT":0}
        revised, _ = a.aggregate_option(self.rows[:3], counts, "quality_disabled")
        self.assertTrue(revised["A"]["eligible"])

    def test_single_positive_entropy_cancellation(self):
        rows = [dict(r, quality_weight=.2 if i==0 else 0) for i,r in enumerate(self.rows[:3])]
        counts = {p.sc.pair_key(r): {"SUPPORT":6,"CONTRADICT":3,"IRRELEVANT":1} for r in rows}
        results, _ = a.aggregate_option(rows, counts, "original_quality")
        self.assertAlmostEqual(results["C"]["score"], .3, delta=1e-12)
        self.assertAlmostEqual(results["D"]["score"], .3, delta=1e-12)

    def test_synthetic_entropy_score_behavior_not_clinical(self):
        rows = self.rows[:2]
        counts = {p.sc.pair_key(rows[0]): {"SUPPORT":4,"CONTRADICT":2,"IRRELEVANT":4},
                  p.sc.pair_key(rows[1]): {"SUPPORT":0,"CONTRADICT":1,"IRRELEVANT":9}}
        results, _ = a.aggregate_option(rows, counts, "quality_disabled")
        self.assertGreater(results["C"]["score"], 0)
        self.assertLess(results["D"]["score"], 0)

    def test_exact_cache_identity_rejects_wrong_seed(self):
        item = p.expected_inventory(self.rows)[0]
        wrong = dict(item["context"], seed=102)
        entry = {"context": wrong, "stance": "SUPPORT", "generation_attempts":1, "attempts":[{"index":1,"status":"success"}]}
        target = self.state / "cache" / (item["cache_key"] + ".json")
        target.parent.mkdir()
        target.write_text(p.sc.canonical(entry), encoding="utf-8")
        with self.assertRaises(ValueError): p.scan_state(ROOT, self.state)

    def test_resumption_skips_valid_samples_and_roundtrip_backup(self):
        first = FakeBackend()
        p.run_missing(ROOT, self.state, self.drive, max_new=2, test_backend=first)
        self.assertEqual(first.calls, 2)
        restored = self.base / "restored"
        p.prepare_state(ROOT, restored, self.drive)
        self.assertEqual(p.scan_state(ROOT, restored)[4]["valid"], 2)
        second = FakeBackend()
        p.run_missing(ROOT, restored, self.drive, max_new=1, test_backend=second)
        self.assertEqual(second.calls, 1)
        self.assertEqual(p.scan_state(ROOT, restored)[4]["valid"], 3)
        self.assertEqual(second.requests[0][1], {"temperature": .7, "seed": 103})

    def test_backup_failure_prevents_model_call(self):
        backend = FakeBackend()
        with patch.object(p.shutil, "copyfileobj", side_effect=IOError("synthetic Drive error")):
            with self.assertRaises(IOError): p.run_missing(ROOT, self.state, self.drive, max_new=1, test_backend=backend)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(p.scan_state(ROOT, self.state)[4]["valid"], 0)

    def test_mounted_storage_without_fsync_still_requires_verified_readback(self):
        with patch.object(p.os, "fsync", side_effect=OSError(p.errno.ENOTSUP, "synthetic unsupported fsync")):
            receipt = p.export_backup(ROOT, self.state, self.base / "exports", self.drive / "checkpoints")
        self.assertFalse(receipt["fsync_supported"])
        self.assertTrue(receipt["backup_verified"])
        self.assertEqual(p.digest(receipt["path"]), receipt["sha256"])

    def test_interruption_requires_explicit_targeted_retry(self):
        backend = FakeBackend(interrupt=True)
        with self.assertRaises(KeyboardInterrupt): p.run_missing(ROOT, self.state, self.drive, max_new=1, test_backend=backend)
        restored = self.base / "restart"
        p.prepare_state(ROOT, restored, self.drive)
        inventory = p.scan_state(ROOT, restored)[3]
        interrupted = [i for i in inventory if i["status"] == "INTERRUPTED"]
        self.assertEqual(len(interrupted), 1)
        second = FakeBackend()
        with self.assertRaises(RuntimeError): p.run_missing(ROOT, restored, self.drive, max_new=1, test_backend=second)
        self.assertEqual(second.calls, 0)
        p.run_missing(ROOT, restored, self.drive, max_new=1, retry_keys=[interrupted[0]["cache_key"]], test_backend=second)
        self.assertEqual(second.calls, 1)
        self.assertEqual(p.scan_state(ROOT, restored)[4]["interrupted"], 0)

    def test_valid_cache_survives_interruption_before_postcall_bookkeeping(self):
        original_write = p.sc.write_json
        def interrupted_write(path, value):
            if Path(path).parent.name == "attempts" and value.get("status") == "COMPLETE":
                raise KeyboardInterrupt("synthetic crash after valid cache commit")
            return original_write(path, value)
        first = FakeBackend()
        with patch.object(p.sc, "write_json", side_effect=interrupted_write):
            with self.assertRaises(KeyboardInterrupt):
                p.run_missing(ROOT, self.state, self.drive, max_new=1, test_backend=first)
        self.assertEqual(first.calls, 1)
        self.assertEqual(p.scan_state(ROOT, self.state)[4]["valid"], 1)
        self.assertEqual(len(p.read_json(self.state / "pilot_state.json")["active_intents"]), 1)
        restored = self.base / "known-result-restart"
        p.prepare_state(ROOT, restored, self.drive)
        self.assertEqual(p.read_json(restored / "pilot_state.json")["active_intents"], {})
        next_backend = FakeBackend()
        p.run_missing(ROOT, restored, self.drive, max_new=1, test_backend=next_backend)
        self.assertEqual(next_backend.calls, 1)
        self.assertEqual(next_backend.requests[0][1]["seed"], 102)

    def test_exhausted_failure_targeted_retry_keeps_audits(self):
        failed = FakeBackend(raw="synthetic invalid response")
        p.run_missing(ROOT, self.state, self.drive, max_new=1, test_backend=failed)
        self.assertEqual(failed.calls, 3)
        inventory = p.scan_state(ROOT, self.state)[3]
        unit = next(i for i in inventory if i["status"] == "FAILED")
        recovered = FakeBackend()
        p.run_missing(ROOT, self.state, self.drive, max_new=1, retry_keys=[unit["cache_key"]], test_backend=recovered)
        self.assertEqual(recovered.calls, 1)
        status = p.scan_state(ROOT, self.state)[4]
        self.assertEqual((status["valid"], status["failed"]), (1,0))
        self.assertTrue((self.state / "cache/failures" / unit["cache_key"] / "0001.json").exists())
        self.assertTrue((self.state / "cache/attempts/history" / unit["cache_key"] / "0001.json").exists())

    def test_targeted_retry_never_calls_unrelated_missing_seed(self):
        item = p.expected_inventory(self.rows)[4]  # seed105, while seed101 is still missing
        p.sc.sample_one(p.projection(item["row"]), item["seed"], FakeBackend(raw="invalid synthetic response"), self.state / "cache")
        backend = FakeBackend()
        p.run_missing(ROOT, self.state, self.drive, max_new=1, retry_keys=[item["cache_key"]], test_backend=backend)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(backend.requests[0][1]["seed"], 105)
        inventory = p.scan_state(ROOT, self.state)[3]
        self.assertEqual(inventory[0]["status"], "MISSING")
        self.assertEqual(inventory[4]["status"], "VALID")

    def test_checkpoint_hash_and_manifest_tamper_rejected(self):
        receipt = p.export_backup(ROOT, self.state, self.base / "exports", self.drive / "checkpoints")
        with self.assertRaises(ValueError): p.inspect_checkpoint(receipt["path"], ROOT, "0"*64)
        altered = self.base / "altered.zip"
        with zipfile.ZipFile(receipt["path"]) as source, zipfile.ZipFile(altered, "w") as target:
            for name in source.namelist():
                data = source.read(name)
                if name == "state/pilot_state.json": data += b" "
                target.writestr(name, data)
        with self.assertRaises(ValueError): p.inspect_checkpoint(altered, ROOT)

    def test_stale_lock_requires_explicit_acknowledgment(self):
        (self.drive / "pilot_run.lock").write_text('{"token":"synthetic-stale"}', encoding="utf-8")
        backend = FakeBackend()
        with self.assertRaises(RuntimeError): p.run_missing(ROOT, self.state, self.drive, max_new=1, test_backend=backend)
        self.assertEqual(backend.calls, 0)
        p.run_missing(ROOT, self.state, self.drive, max_new=1, test_backend=backend, acknowledge_stale_lock=True)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(len(list((self.drive / "checkpoints").glob("stale_lock_*.json"))), 1)

    def test_local_real_backend_blocked_without_call(self):
        with patch.object(p.sc, "verify_ollama_identity") as api:
            with self.assertRaises(RuntimeError): p.run_missing(ROOT, self.state, self.drive, max_new=1)
            api.assert_not_called()
        with self.assertRaises(RuntimeError): p.setup_ollama(ROOT, self.state)

    def test_incomplete_checkpoint_never_opens_gold(self):
        receipt = p.export_backup(ROOT, self.state, self.base / "exports", self.drive / "checkpoints")
        with self.assertRaises(ValueError):
            p.evaluate_checkpoint(ROOT, receipt["path"], receipt["sha256"], self.base / "does-not-exist-gold.jsonl", self.base / "eval")
        self.assertFalse((self.base / "eval").exists())

    def test_complete_cache_is_not_final_without_live_identity(self):
        for item in p.expected_inventory(self.rows):
            path = self.state / "cache" / (item["cache_key"] + ".json")
            path.parent.mkdir(exist_ok=True)
            path.write_text(p.sc.canonical({"context":item["context"], "stance":"IRRELEVANT", "generation_attempts":1, "attempts":[{"index":1,"status":"success"}]}), encoding="utf-8")
        self.assertTrue(p.scan_state(ROOT, self.state)[4]["complete_samples"])
        with self.assertRaises(RuntimeError): p.finish(ROOT, self.state, self.drive)

    def test_wrong_stored_live_model_digest_is_rejected(self):
        meta = p.read_json(self.state / "pilot_state.json")
        meta["model_identity"] = {"model":p.sc.MODEL, "digest":"0"*64, "ollama_version":p.sc.OLLAMA_VERSION,
            "verified_live_identity":True, "backend":"unchanged OllamaTextGenerationBackend", "thinking":True, "temperature":.7, "size_vram":1}
        p.save_meta(self.state, meta)
        with self.assertRaises(ValueError): p.scan_state(ROOT, self.state)

    def test_final_roundtrip_with_explicit_synthetic_identity_fixture(self):
        # Simulated API metadata and synthetic labels, ONLY in this temporary
        # directory. This is a packaging test, not a real completed pilot.
        meta = p.read_json(self.state / "pilot_state.json")
        meta["model_identity"] = {"model":p.sc.MODEL, "digest":p.sc.DIGEST, "ollama_version":p.sc.OLLAMA_VERSION,
            "verified_live_identity":True, "backend":"unchanged OllamaTextGenerationBackend", "thinking":True, "temperature":.7, "size_vram":1}
        p.save_meta(self.state, meta)
        for item in p.expected_inventory(self.rows):
            target = self.state / "cache" / (item["cache_key"] + ".json")
            target.parent.mkdir(exist_ok=True)
            target.write_text(p.sc.canonical({"context":item["context"], "stance":"IRRELEVANT", "generation_attempts":1, "attempts":[{"index":1,"status":"success"}]}), encoding="utf-8")
        receipt = p.finish(ROOT, self.state, self.drive)
        self.assertTrue(receipt["backup_verified"])
        m, _ = p.inspect_checkpoint(receipt["path"], ROOT, receipt["sha256"])
        self.assertTrue(m["final"])
        self.assertEqual(m["status"]["valid"], 150)
        self.assertEqual(m["status"]["recorded_generation_attempts"], 150)
        restored = self.base / "final-restored"
        p.restore_state(receipt["path"], ROOT, restored, receipt["sha256"])
        backend = FakeBackend()
        status = p.run_missing(ROOT, restored, self.drive, test_backend=backend)
        self.assertEqual(backend.calls, 0)
        self.assertTrue(status["complete_samples"])

    def test_gold_opened_only_after_blind_prediction_write(self):
        # Evaluator ordering unit test uses a MOCK checkpoint and SYNTHETIC gold
        # inside a temporary directory; it does not produce pilot results.
        inventory = p.expected_inventory(self.rows)
        for item in inventory: item["entry"] = {"stance":"IRRELEVANT"}
        gold = self.base / "synthetic-gold.jsonl"
        gold.write_text(json.dumps({"id":p.QID, "question":self.rows[0]["question_stem"],
            "options":{o: next(r["candidate_option_text"] for r in self.rows if r["candidate_option_id"]==o) for o in a.OPTIONS}, "answer":"A"}) + "\n", encoding="utf-8")
        output = self.base / "mock-evaluation"
        real_open = Path.open
        observed = []
        def checked_open(path, *args, **kwargs):
            if path == gold:
                self.assertTrue((output / "blind_predictions.json").is_file())
                observed.append(True)
            return real_open(path, *args, **kwargs)
        with patch.object(p, "inspect_checkpoint", return_value=({"final":True,"status":{"complete_samples":True}}, {})), \
             patch.object(p, "restore_state"), patch.object(p, "scan_state", return_value=(self.rows,self.sampling,{},inventory,{})), \
             patch.object(Path, "open", checked_open):
            p.evaluate_checkpoint(ROOT, self.base/"mock.zip", "0"*64, gold, output)
        self.assertTrue(observed)
        self.assertTrue(p.read_json(output / "evaluation.json")["gold_joined_after_blind_prediction"])


if __name__ == "__main__":
    unittest.main()

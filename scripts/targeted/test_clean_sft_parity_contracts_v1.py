#!/usr/bin/env python3
"""CPU-only quality, reduction, resume, and cache-isolation tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clean_sft_parity_core_v1 as c
import _clean_sft_parity_kc_v1 as kc


class KnowledgeConditionTests(unittest.TestCase):
    def test_frozen_opd_template_is_byte_identical(self):
        audit = kc.teacher_condition_parity()
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(audit["templates_byte_identical"])
        self.assertEqual(
            audit["opd_template_sha256"],
            audit["sft_matched_prompt_template_sha256"],
        )
        self.assertFalse(
            audit["wording_difference_audit"]["selective_clean"]
            ["distribution_equivalent_claim_allowed"]
        )


class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        self.template = [{
            "target_id": "t1", "source": "源句",
            "canonical_en_target": "Quartz",
            "candidate_translation": "The sample contains Quartz.",
            "review_stratum": "generated_or_repaired",
        }]

    def passing_row(self):
        return [{
            **self.template[0], "review_status": "PASS",
            "faithful_complete_translation": True,
            "terminology_correct": True, "clean_output_only": True,
        }]

    def test_all_strict_fields_are_required(self):
        self.assertEqual(
            kc.validate_manual_review(self.template, self.passing_row())["status"],
            "PASS",
        )
        for field in (
            "faithful_complete_translation", "terminology_correct",
            "clean_output_only",
        ):
            rows = self.passing_row()
            rows[0][field] = None
            self.assertEqual(
                kc.validate_manual_review(self.template, rows)["status"],
                "FAIL",
            )

    def test_replacement_and_duplicates_are_rejected(self):
        self.assertEqual(
            kc.validate_manual_review(self.template, [])["status"],
            "FAIL",
        )
        replaced = self.passing_row()
        replaced[0]["candidate_translation"] = "replacement"
        self.assertEqual(
            kc.validate_manual_review(self.template, replaced)["status"],
            "FAIL",
        )
        duplicate = self.passing_row() * 2
        self.assertEqual(
            kc.validate_manual_review(self.template, duplicate)["status"],
            "FAIL",
        )


class TokenReductionTests(unittest.TestCase):
    def test_accumulated_ce_sum_divided_by_all_tokens_matches_global_mean(self):
        import torch
        import torch.nn.functional as functional

        torch.manual_seed(7)
        global_model = torch.nn.Linear(4, 5, bias=False, dtype=torch.float64)
        accum_model = torch.nn.Linear(4, 5, bias=False, dtype=torch.float64)
        accum_model.load_state_dict(global_model.state_dict())
        micros = [
            (torch.randn(3, 4, dtype=torch.float64), torch.tensor([0, 1, 2])),
            (torch.randn(2, 4, dtype=torch.float64), torch.tensor([3, 4])),
            (torch.randn(4, 4, dtype=torch.float64), torch.tensor([1, 1, 0, 2])),
        ]
        inputs = torch.cat([x for x, _ in micros])
        labels = torch.cat([y for _, y in micros])
        functional.cross_entropy(global_model(inputs), labels, reduction="mean").backward()
        token_count = 0
        for inputs_micro, labels_micro in micros:
            loss_sum = functional.cross_entropy(
                accum_model(inputs_micro), labels_micro, reduction="sum"
            )
            loss_sum.backward()
            token_count += labels_micro.numel()
        for parameter in accum_model.parameters():
            parameter.grad.div_(token_count)
        self.assertTrue(torch.allclose(
            global_model.weight.grad, accum_model.weight.grad,
            atol=1e-12, rtol=1e-12,
        ))


class IsolationTests(unittest.TestCase):
    def test_immutable_contract_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.json"
            first = kc.ensure_immutable_contract(path, {"arm": "a", "lr": 2e-5})
            self.assertEqual(
                first, kc.ensure_immutable_contract(path, {"arm": "a", "lr": 2e-5})
            )
            with self.assertRaisesRegex(RuntimeError, "ARTIFACT_IDENTITY_MISMATCH"):
                kc.ensure_immutable_contract(path, {"arm": "a", "lr": 3e-5})

    def test_interrupted_checkpoint_recovers_latest_complete_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for epoch in (1, 2):
                checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:03d}"
                (checkpoint / "hf").mkdir(parents=True)
                (checkpoint / "state.pt").write_bytes(b"state")
                c.atomic_json(checkpoint / "meta.json", {
                    "status": "PASS", "completed_epoch": epoch,
                    "completed_updates": epoch * 344,
                    "run_contract_sha256": "contract-a",
                })
            incomplete = run_dir / "checkpoints" / "epoch_003.tmp"
            incomplete.mkdir(parents=True)
            latest = kc.resolve_latest_checkpoint(run_dir, "contract-a")
            self.assertEqual(latest.name, "epoch_002")
            self.assertEqual(
                c.read_json(run_dir / "latest_resume.json")["completed_epoch"], 2
            )
            with self.assertRaisesRegex(RuntimeError, "RESUME_CONTRACT_MISMATCH"):
                kc.resolve_latest_checkpoint(run_dir, "contract-b")

    def test_eval_cache_is_bound_to_contract_and_unique_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            c.atomic_jsonl(path, [{
                "status": "PASS", "eval_id": "e1",
                "eval_contract_sha256": "eval-a",
            }])
            self.assertEqual(set(kc.pass_map(path, "eval_id", "eval-a")), {"e1"})
            with self.assertRaisesRegex(RuntimeError, "EVAL_CACHE_IDENTITY_MISMATCH"):
                kc.pass_map(path, "eval_id", "eval-b")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "status": "PASS", "eval_id": "e1",
                    "eval_contract_sha256": "eval-a",
                }) + "\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate cached prediction"):
                kc.pass_map(path, "eval_id", "eval-a")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""CPU-only regression tests for KC-SFT v2 freeze and recovery contracts."""
from __future__ import annotations

import argparse
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clean_sft_checkpoint_v2 as checkpoint_v2
import _clean_sft_materialize_v2 as materialize_v2
import _clean_sft_optimization_contract_v2 as optimization_v2
import _clean_sft_parity_core_v1 as c
import _clean_sft_parity_kc_v1 as kc
import clean_sft_parity_runtime_v1 as runtime


class RawDistributionAdmissionTests(unittest.TestCase):
    def bad_content_record(self, dataset: str) -> dict:
        audit = c.audit_translation(
            "The designated chemistry term was omitted.", "Quartz"
        )
        classification = materialize_v2.classify_output(
            dataset, "The designated chemistry term was omitted.", audit, 42
        )
        return {"target_id": "t1", **classification}

    def test_matched_prompt_preserves_and_admits_content_failure(self):
        record = self.bad_content_record("matched_prompt")
        self.assertEqual(record["status"], "RAW_OUTPUT_RECORDED")
        self.assertIn("canonical_term_missing", record["content_quality_flags"])
        self.assertTrue(record["technical_valid"])
        raw_text = "The designated chemistry term was omitted."
        stored = {
            **record, "target_translation": raw_text,
            "teacher_prompt_sha256": "prompt-a",
            "generation_contract_sha256": "contract-a",
        }
        candidate = materialize_v2._build_candidate_rows(
            "matched_prompt",
            [{"target_id": "t1", "target_translation": "old", "reference": "old"}],
            {"t1"}, {"t1": stored},
        )[0]
        self.assertEqual(candidate["target_translation"], raw_text)
        self.assertTrue(candidate["clean_sft_provenance"]["raw_output_preserved_without_replacement"])
        admission = materialize_v2.summarize_admission(
            "matched_prompt", [record], {"t1"}, "PASS"
        )
        self.assertTrue(admission["admitted"])
        self.assertEqual(admission["content_quality_status"], "OBSERVED_NOT_GATED")
        self.assertEqual(admission["content_quality_flagged_count"], 1)
        self.assertFalse(admission["all_targets_semantically_correct"])

    def test_selective_clean_still_blocks_content_failure(self):
        record = self.bad_content_record("selective_clean")
        self.assertEqual(record["status"], "CONTENT_REJECTED")
        admission = materialize_v2.summarize_admission(
            "selective_clean", [record], {"t1"}, "PASS"
        )
        self.assertFalse(admission["admitted"])
        self.assertEqual(admission["content_quality_status"], "FAIL")

    def test_empty_and_overlength_are_technical_not_content_gates(self):
        empty = materialize_v2.classify_output(
            "matched_prompt", "", c.audit_translation("", "Quartz"), 3
        )
        self.assertEqual(empty["status"], "TECHNICAL_INVALID")
        self.assertIn("empty_output", empty["technical_flags"])
        long = materialize_v2.classify_output(
            "matched_prompt", "Quartz", c.audit_translation("Quartz", "Quartz"),
            c.MAX_TOTAL_LENGTH + 1,
        )
        self.assertIn("max_total_length_exceeded", long["technical_flags"])


class MaterializationIsolationTests(unittest.TestCase):
    def test_audit_only_does_not_touch_candidate_or_history(self):
        context = {
            "job_id": 0, "src_text": "源句", "src_term": "石英",
            "lexical_record": {"en_name": "Quartz", "en_alias": ""},
        }
        old = {
            "job_id": 0, "target_id": "t1",
            "target_translation": "Quartz.",
        }
        args = argparse.Namespace(
            dataset="matched_prompt", finalize=False, audit_only=True,
            max_new_tokens=384, max_retries=3, device="npu:0",
            review_sample_size=100, manual_review_jsonl=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            candidate_manifest = directory / "candidate_manifest.json"
            candidate = directory / "candidate" / "candidate.jsonl"
            attempts = directory / "candidate" / "regeneration_attempts.jsonl"
            c.atomic_json(candidate_manifest, {"sentinel": "manifest"})
            c.atomic_jsonl(candidate, [{"sentinel": "candidate"}])
            c.atomic_jsonl(attempts, [{"sentinel": "history"}])
            before = {
                path: path.read_bytes()
                for path in (candidate_manifest, candidate, attempts)
            }
            with mock.patch.dict(
                kc.DATASETS["matched_prompt"],
                {"directory": directory}, clear=False,
            ), mock.patch.object(
                materialize_v2.c, "read_jsonl", side_effect=[[context], [old]]
            ), mock.patch.object(
                materialize_v2.c, "source_identity",
                return_value={"status": "PASS", "issues": [], "rows": 1},
            ):
                materialize_v2.cmd_materialize(args)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_decoding_change_cannot_resume_old_generation(self):
        fake_teacher = {"identity_sha256": "teacher-a", "files": []}
        first = materialize_v2.build_generation_contract(
            "matched_prompt", 384, 3, fake_teacher
        )
        changed = materialize_v2.build_generation_contract(
            "matched_prompt", 512, 3, fake_teacher
        )
        self.assertNotEqual(c.sha256_json(first), c.sha256_json(changed))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generation_contract.json"
            digest = kc.ensure_immutable_contract(path, first)
            with self.assertRaisesRegex(RuntimeError, "ARTIFACT_IDENTITY_MISMATCH"):
                kc.ensure_immutable_contract(path, changed)
            history = [{
                "target_id": "t1", "variant": 0,
                "generation_contract_sha256": digest,
                "technical_valid": True, "status": "RAW_OUTPUT_RECORDED",
            }]
            with self.assertRaisesRegex(RuntimeError, "GENERATION_CONTRACT_MISMATCH"):
                materialize_v2.inspect_history(
                    history, "matched_prompt", "changed", {"t1"}, 1
                )

    def test_completed_matched_target_is_not_scheduled_again(self):
        rows = [{
            "target_id": "t1", "variant": 0,
            "generation_contract_sha256": "contract-a",
            "technical_valid": True, "status": "RAW_OUTPUT_RECORDED",
        }]
        completed, next_variant = materialize_v2.inspect_history(
            rows, "matched_prompt", "contract-a", {"t1"}, 1
        )
        self.assertEqual(set(completed), {"t1"})
        self.assertEqual(next_variant, {})


class CheckpointRecoveryTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        import torch
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_server_torch_restores_model_optimizer_scheduler_and_rng(self):
        import torch

        torch.manual_seed(19)
        random.seed(19)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 - step / 10
        )
        loss = model(torch.ones(2, 3)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        rng = runtime.capture_rng()
        state = {
            "state_format_version": "kc_sft_checkpoint_v2",
            "completed_epoch": 1, "completed_updates": 344,
            "run_contract_sha256": "contract-a",
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "rng": rng,
        }
        pointer = {
            "completed_epoch": 1, "completed_updates": 344,
            "run_contract_sha256": "contract-a",
        }
        meta = {
            **pointer, "status": "PASS",
            "state_format_version": "kc_sft_checkpoint_v2",
        }
        expected_python = random.random()
        expected_torch = torch.rand(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save(state, path)
            loaded = checkpoint_v2.torch_load_compat(path)
        checkpoint_v2.validate_pointer_meta_state(
            pointer, meta, loaded, "contract-a"
        )
        restored_model = torch.nn.Linear(3, 2)
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(), lr=9e-4
        )
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
            restored_optimizer, lr_lambda=lambda step: 1.0 - step / 10
        )
        restored_model.load_state_dict(loaded["model"])
        restored_optimizer.load_state_dict(loaded["optimizer"])
        restored_scheduler.load_state_dict(loaded["scheduler"])
        runtime.restore_rng(loaded["rng"])
        self.assert_nested_equal(model.state_dict(), restored_model.state_dict())
        self.assert_nested_equal(
            optimizer.state_dict(), restored_optimizer.state_dict()
        )
        self.assert_nested_equal(
            scheduler.state_dict(), restored_scheduler.state_dict()
        )
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))

    def test_checkpoint_completion_matches_update_ledger(self):
        records = [
            {"optimizer_update": update, "epoch": 1}
            for update in range(1, 345)
        ]
        committed = checkpoint_v2.committed_metric_prefix(records, 344, 1)
        self.assertEqual(len(committed), 344)
        with self.assertRaisesRegex(RuntimeError, "ledger mismatch"):
            checkpoint_v2.committed_metric_prefix(records[:-1], 344, 1)
        duplicate = records[:-1] + [records[-2]]
        with self.assertRaisesRegex(RuntimeError, "ledger mismatch"):
            checkpoint_v2.committed_metric_prefix(duplicate, 344, 1)


class LearningRateContractTests(unittest.TestCase):
    def test_both_learning_rate_contracts_only_differ_in_lr(self):
        low = optimization_v2.build("KC_SFT_MATCHED_PROMPT_LR2E5")
        high = optimization_v2.build("KC_SFT_MATCHED_PROMPT_LR3E5")
        self.assertEqual(low["lr"], 2e-5)
        self.assertEqual(high["lr"], 3e-5)
        self.assertEqual(low["ce_compute_dtype"], "float32")
        self.assertEqual(low["total_updates"], 1032)
        low_without_lr = {k: v for k, v in low.items() if k != "lr"}
        high_without_lr = {k: v for k, v in high.items() if k != "lr"}
        self.assertEqual(low_without_lr, high_without_lr)
        self.assertEqual(
            optimization_v2.build("KC_SFT_SELECTIVE_CLEAN_LR2E5"), low
        )
        self.assertEqual(
            optimization_v2.build("KC_SFT_SELECTIVE_CLEAN_LR3E5"), high
        )
    def test_bfloat16_forward_logits_use_float32_ce(self):
        import torch
        import torch.nn.functional as functional

        logits = torch.randn(7, 11, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        ce_sum = functional.cross_entropy(
            logits.float(), labels, reduction="sum"
        )
        self.assertEqual(ce_sum.dtype, torch.float32)
        ce_sum.backward()
        self.assertTrue(torch.isfinite(logits.grad.float()).all())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""CPU-only contract tests for Chemistry Clean-SFT parity v1."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clean_sft_parity_core_v1 as core
import clean_sft_parity_runtime_v1 as runtime


class TargetAuditTests(unittest.TestCase):
    def test_clean_translation(self):
        audit = core.audit_translation(
            "The sample contained C.I. Basic violet 1 and remained stable.",
            "C.I. Basic violet 1",
        )
        self.assertEqual(audit["flags"], [])
        self.assertTrue(audit["strict_canonical_hit"])

    def test_meta_echo_is_rejected(self):
        audit = core.audit_translation(
            "The designated chemistry term is: Malathion. "
            "The sample was tested.",
            "Malathion",
        )
        self.assertIn("designated_chemistry_term", audit["flags"])

    def test_cjk_is_rejected(self):
        audit = core.audit_translation(
            "The sample 含有 Imiquimod.", "Imiquimod"
        )
        self.assertIn("cjk_residual", audit["flags"])

    def test_ci_spacing_is_supplement_only(self):
        target = "C.I. Pigment Yellow 14"
        output = "C. I. Pigment Yellow 14 was measured."
        self.assertTrue(core.ci_spacing_hit(target, output))
        self.assertFalse(core.strict_hit(target, output))

    def test_epoch_order_is_deterministic_and_distinct(self):
        self.assertEqual(
            core.epoch_order_sha(5500, 1),
            core.epoch_order_sha(5500, 1),
        )
        self.assertNotEqual(
            core.epoch_order_sha(5500, 1),
            core.epoch_order_sha(5500, 2),
        )

    def test_update_budget(self):
        micros = core.math.ceil(5500 / core.MICRO_BATCH)
        updates = core.math.ceil(micros / core.GRAD_ACCUM)
        self.assertEqual(updates, 344)
        self.assertEqual(updates * core.EPOCHS, 1032)

    def test_score_targeted_slices(self):
        rows = []
        for index in range(1000):
            rows.append({
                "eval_domain": "chemistry",
                "eval_split": "uc" if index < 500 else "uw",
                "src_term": "测试",
                "canonical_en_target": "Quartz",
                "model_translation": (
                    "Quartz was measured."
                    if index % 2 == 0 else "The material was measured."
                ),
            })
        result = runtime.score_targeted(rows)
        self.assertEqual(result["all"]["strict_canonical"]["n"], 1000)
        self.assertEqual(result["all"]["strict_canonical"]["hits"], 500)
        self.assertEqual(result["uc"]["strict_canonical"]["hits"], 250)
        self.assertEqual(result["uw"]["strict_canonical"]["hits"], 250)


if __name__ == "__main__":
    unittest.main()

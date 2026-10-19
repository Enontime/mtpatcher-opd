#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _targeted_tensorboard_v1 as tb
import tensorboard_bridge_v1 as bridge


class FakeWriter:
    instances = []

    def __init__(self, log_dir, purge_step=None):
        self.log_dir = log_dir
        self.purge_step = purge_step
        self.scalars = []
        self.closed = False
        self.__class__.instances.append(self)

    def add_scalar(self, tag, value, global_step, walltime=None):
        self.scalars.append((tag, float(value), int(global_step), walltime))

    def flush(self):
        pass

    def close(self):
        self.closed = True


class HistoricalTrainingParserTests(unittest.TestCase):
    def write(self, root: Path, text: str) -> Path:
        path = root / "train.log"
        path.write_text(text, encoding="utf-8")
        return path

    def test_opd_fields_keep_window_semantics_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), "\n".join([
                "2026-10-14T22:49:06+08:00 arm=O2 epoch=1/3 update=10/1032 pct=1.0% kl100=1.127147 rollout_len200=27.9 grad_norm=32.25 lr=6.6e-06 ETA=1m",
                "2026-10-14T22:51:30+08:00 arm=O2 epoch=1/3 update=20/1032 pct=1.9% kl100=0.893181 rollout_len200=29.0 grad_norm=15.3125 lr=1.3e-05 ETA=1m",
            ]))
            rows, report = bridge.parse_training_log(path, max_grad_norm=1.0)
        by_tag = {}
        for row in rows:
            by_tag.setdefault(row.tag, []).append(row)
        self.assertEqual([row.step for row in by_tag["opd/kl_recent100"]], [10, 20])
        self.assertEqual(by_tag["opd/kl_recent100"][0].source_field, "kl100")
        self.assertEqual(by_tag["train/rollout_length_recent200"][0].value, 27.9)
        self.assertAlmostEqual(by_tag["optim/grad_clip_scale"][0].value, 1.0 / 32.25)
        self.assertEqual(report["last_logged_step"], 20)

    def test_historical_sft_loss_is_not_renamed_exact_token_ce(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), "2026-09-10T23:49:21+08:00 arm=C2 epoch=1/3 update=10/1032 pct=1.0% loss50=1.436818 grad_norm=23.25 lr=6.6e-06 ETA=1m\n")
            rows, _ = bridge.parse_training_log(path, max_grad_norm=1.0)
        tags = {row.tag for row in rows}
        self.assertIn("sft/ce_microbatch_mean_recent50", tags)
        self.assertNotIn("sft/ce_per_response_token", tags)

    def test_non_monotonic_source_steps_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), "\n".join([
                "2026-01-01T00:00:00+08:00 arm=O2 epoch=1/3 update=20/30 kl100=1 rollout_len200=2 grad_norm=3 lr=4e-6",
                "2026-01-01T00:00:01+08:00 arm=O2 epoch=1/3 update=10/30 kl100=1 rollout_len200=2 grad_norm=3 lr=4e-6",
            ]))
            with self.assertRaisesRegex(RuntimeError, "non-monotonic optimizer update"):
                bridge.parse_training_log(path, max_grad_norm=1.0)

    def test_malformed_numeric_field_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), "2026-01-01T00:00:00+08:00 arm=O2 epoch=1/3 update=10/30 kl100=bad rollout_len200=2 grad_norm=3 lr=4e-6\n")
            with self.assertRaisesRegex(RuntimeError, "invalid log field kl100"):
                bridge.parse_training_log(path, max_grad_norm=1.0)

    def test_source_log_is_never_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), "2026-01-01T00:00:00+08:00 arm=C2 epoch=1/3 update=10/30 loss50=1 grad_norm=3 lr=4e-6\n")
            before = tb.sha256_file(path)
            bridge.parse_training_log(path, max_grad_norm=1.0)
            self.assertEqual(tb.sha256_file(path), before)


class NativeWriterTests(unittest.TestCase):
    def setUp(self):
        FakeWriter.instances.clear()

    def test_resume_uses_purge_step_and_rejects_rolled_back_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = tb.NativeTensorBoard(
                enabled=True, log_dir=Path(tmp), run_id="run",
                durable_step=344, write_every=1, writer_class=FakeWriter,
            )
            self.assertEqual(FakeWriter.instances[-1].purge_step, 345)
            with self.assertRaisesRegex(RuntimeError, "not after durable"):
                writer.add_scalars(344, {"optim/lr": 1e-5})
            writer.log_sft_update(
                step=345, ce_per_response_token=0.4, lr=1e-5,
                grad_norm_preclip=2.0, max_grad_norm=1.0,
                response_tokens=99, step_seconds=0.2,
            )
            self.assertIn(("optim/grad_clip_scale", 0.5, 345, None), FakeWriter.instances[-1].scalars)

    def test_disabled_writer_does_not_import_or_create_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "absent"
            writer = tb.NativeTensorBoard(
                enabled=False, log_dir=path, run_id="run",
                durable_step=0, writer_class=FakeWriter,
            )
            writer.add_scalars(1, {"optim/lr": 1e-5})
            self.assertFalse(path.exists())
            self.assertEqual(FakeWriter.instances, [])


    def test_logdir_cannot_be_reused_by_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            first = tb.NativeTensorBoard(
                enabled=True, log_dir=path, run_id="run-a",
                durable_step=0, writer_class=FakeWriter,
            )
            first.close()
            with self.assertRaisesRegex(RuntimeError, "contract changed"):
                tb.NativeTensorBoard(
                    enabled=True, log_dir=path, run_id="run-a",
                    durable_step=0, write_every=2, writer_class=FakeWriter,
                )
            with self.assertRaisesRegex(RuntimeError, "another run"):
                tb.NativeTensorBoard(
                    enabled=True, log_dir=path, run_id="run-b",
                    durable_step=0, writer_class=FakeWriter,
                )


class ConversionAndEvaluationTests(unittest.TestCase):
    def setUp(self):
        FakeWriter.instances.clear()

    def test_metric_names_keep_incompatible_units_separate(self):
        self.assertEqual(
            tb.METRIC_DEFINITIONS["opd/kl_recent100"]["unit"],
            "nats_per_response_token_row_mean",
        )
        self.assertEqual(
            tb.METRIC_DEFINITIONS["sft/ce_microbatch_mean_recent50"]["unit"],
            "nats_per_token_microbatch_mean",
        )
        self.assertNotEqual(
            tb.METRIC_DEFINITIONS["opd/kl_recent100"]["definition"],
            tb.METRIC_DEFINITIONS["sft/ce_per_response_token"]["definition"],
        )

    def test_chemistry_preset_covers_all_required_learning_rates(self):
        self.assertTrue({
            "opd_lr1e6_horizon5", "opd_lr1e5", "opd_lr2e5", "opd_lr3e5",
            "sft_c2_old", "sft_a1_horizon5", "kc_sft_matched_prompt_lr2e5",
        }.issubset(bridge.PRESETS["chemistry"]))

    def test_content_addressed_repeat_conversion_is_idempotent(self):
        plan = {
            "status": "READY", "run_id": "fixture", "method": "opd",
            "conversion_identity_sha256": "a" * 64,
            "records": [tb.ScalarRecord("optim/lr", 10, 1e-5).to_json()],
            "scalar_counts": {"optim/lr": 1}, "metric_definitions": {},
            "sources": [], "missing": [], "evaluation_provenance": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            first, first_status = bridge.write_plan(plan, Path(tmp), FakeWriter)
            second, second_status = bridge.write_plan(plan, Path(tmp), FakeWriter)
            self.assertEqual(first, second)
            self.assertEqual(first_status, "CREATED")
            self.assertEqual(second_status, "ALREADY_CURRENT")
            self.assertEqual(len(FakeWriter.instances), 1)

    def test_ci_and_non_ci_use_frozen_summary_groups(self):
        summary = {
            "CI110": {"new_accuracy": 0.7},
            "NONCI890": {"new_accuracy": 0.2},
        }
        rows = bridge.slice_records(summary, step=1032, path=Path("slice.json"), eval_id="e")
        values = {row.tag: row.value for row in rows}
        self.assertEqual(values["knowledge/chemistry_ci"], 0.7)
        self.assertEqual(values["knowledge/chemistry_non_ci"], 0.2)

    def test_clean_sft_nested_slices_preserve_frozen_groups(self):
        summary = {"slices": {
            "all": {
                "strict_canonical": {"accuracy": 0.31},
                "ci_spacing_normalized": {"accuracy": 0.34},
            },
            "ci": {"strict_canonical": {"accuracy": 0.7}},
            "non_ci": {"strict_canonical": {"accuracy": 0.2}},
            "uc": {"strict_canonical": {"accuracy": 0.4}},
            "uw": {"strict_canonical": {"accuracy": 0.22}},
        }}
        rows = bridge.chemistry_records(
            summary, step=10, path=Path("chemistry.json"), eval_id="e",
        )
        values = {row.tag: row.value for row in rows}
        self.assertEqual(values["knowledge/chemistry_strict"], 0.31)
        self.assertEqual(values["knowledge/chemistry_spacing_normalized"], 0.34)
        self.assertEqual(values["knowledge/chemistry_ci"], 0.7)
        self.assertEqual(values["knowledge/chemistry_non_ci"], 0.2)

    def test_spacing_normalized_is_absent_when_source_does_not_have_it(self):
        rows = bridge.chemistry_records(
            {"overall": {"accuracy": 0.2}},
            step=1, path=Path("chemistry.json"), eval_id="e",
        )
        self.assertNotIn("knowledge/chemistry_spacing_normalized", {row.tag for row in rows})

    def test_opd_horizon_uses_snapshot_checkpoint_and_optimizer_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm_dir = root / "O2"
            checkpoint = arm_dir / "train" / "epoch_checkpoints" / "pass_1_hf"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            summary_path = root / "learning_curve" / "heldout" / "O2" / "P1" / "chemistry_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps({
                "metric": "strict fixture",
                "overall": {"accuracy": 0.2},
                "uc500": {"accuracy": 0.3},
                "uw500": {"accuracy": 0.1},
            }), encoding="utf-8")
            logs = root / "logs"
            logs.mkdir()
            snapshot = logs / "O2_snapshot.log"
            snapshot.write_text(
                f"arm=O2 pass=1 SNAPSHOT_PASS path={checkpoint}\n",
                encoding="utf-8",
            )
            manifest_path = arm_dir / "train" / "manifest.json"
            manifest = {"training": {"updates_per_epoch": 10, "epochs": 1}}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            experiment = bridge.Experiment(
                run_id="horizon", root=root, arm="O2", arm_dir=arm_dir,
                manifest_path=manifest_path, training_log=None, method="opd",
            )
            rows, provenance, _, missing = bridge.collect_opd_horizon_passes(
                experiment, manifest,
            )
            self.assertEqual({row.step for row in rows}, {10})
            self.assertEqual(provenance[0]["optimizer_step"], 10)
            self.assertIn("per-pass quality", missing[0])

    def test_eval_index_requires_sha_checkpoint_and_gain_loss_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "eval.json"
            source.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
            checkpoint_identity = bridge.model_identity(checkpoint)["identity_sha256"]
            index = root / "index.json"
            index.write_text(json.dumps({"eval_records": [{
                "run_id": "r", "eval_id": "frozen-eval", "optimizer_step": 10,
                "checkpoint_path": str(checkpoint),
                "checkpoint_identity_sha256": checkpoint_identity,
                "eval_data_sha256": tb.sha256_file(source),
                "prompt_sha256": "prompt-sha",
                "decoding_config_sha256": "decoding-sha",
                "evaluation_version": "fixture-v1",
                "source_files": [{"path": str(source), "sha256": tb.sha256_file(source)}],
                "metrics": {"knowledge/gain": 2, "knowledge/loss": 1},
                "baseline": {"eval_id": "frozen-eval", "model_identity_sha256": "baseline-sha"},
            }]}), encoding="utf-8")
            rows, provenance, _ = bridge.collect_eval_index(index, "r")
            self.assertEqual({row.tag for row in rows}, {"knowledge/gain", "knowledge/loss"})
            self.assertEqual(provenance[0]["baseline"]["model_identity_sha256"], "baseline-sha")
            payload = json.loads(index.read_text(encoding="utf-8"))
            payload["eval_records"][0]["source_files"][0]["sha256"] = "wrong"
            index.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "evaluation SHA mismatch"):
                bridge.collect_eval_index(index, "r")

    def test_bound_eval_contract_validates_checkpoint_inputs_and_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm = "KC_SFT_MATCHED_PROMPT_LR2E5"
            arm_dir = root / arm
            train_dir = arm_dir / "train"
            eval_dir = arm_dir / "eval"
            final_model = train_dir / "final_hf"
            checkpoint = train_dir / "checkpoints" / "epoch_003"
            eval_dir.mkdir(parents=True)

            def write_model(path: Path) -> list[dict]:
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}\n", encoding="utf-8")
                (path / "model.safetensors").write_bytes(b"fixture weights")
                return [{
                    "name": name,
                    "size": (path / name).stat().st_size,
                    "sha256": tb.sha256_file(path / name),
                } for name in ("config.json", "model.safetensors")]

            declared_files = write_model(final_model)
            write_model(checkpoint / "hf")
            run_contract = {"arm": arm, "version": "fixture"}
            run_contract_sha = tb.sha256_json(run_contract)
            (train_dir / "run_contract.json").write_text(
                json.dumps({
                    "status": "FROZEN",
                    "contract": run_contract,
                    "contract_sha256": run_contract_sha,
                }), encoding="utf-8",
            )
            (train_dir / "latest_resume.json").write_text(json.dumps({
                "checkpoint": str(checkpoint),
                "completed_updates": 1032,
                "run_contract_sha256": run_contract_sha,
            }), encoding="utf-8")
            (checkpoint / "meta.json").write_text(json.dumps({
                "status": "PASS",
                "completed_updates": 1032,
                "run_contract_sha256": run_contract_sha,
            }), encoding="utf-8")
            train_manifest = {
                "arm": arm,
                "final_model": str(final_model),
                "run_contract_sha256": run_contract_sha,
                "training": {"total_updates": 1032},
            }
            manifest_path = train_dir / "manifest.json"
            manifest_path.write_text(json.dumps(train_manifest), encoding="utf-8")

            targeted_input = root / "targeted.jsonl"
            general_input = root / "wmt24.jsonl"
            targeted_input.write_text("{}\n", encoding="utf-8")
            general_input.write_text("{}\n", encoding="utf-8")
            contract = {
                "arm": arm,
                "evaluation_version": "fixture-eval-v1",
                "generation": {"prompt_sha256": "prompt", "do_sample": False},
                "inputs": {
                    "targeted": {
                        "path": str(targeted_input),
                        "sha256": tb.sha256_file(targeted_input),
                    },
                    "general": {"WMT24": {
                        "path": str(general_input),
                        "sha256": tb.sha256_file(general_input),
                    }},
                },
                "model": {
                    "resolved_path": str(final_model),
                    "files": declared_files,
                    "identity_sha256": tb.sha256_json(declared_files),
                },
                "scoring": {"targeted": "strict-v1", "general": "bleu-v1"},
            }
            contract_sha = tb.sha256_json(contract)
            (eval_dir / "eval_contract.json").write_text(json.dumps({
                "status": "BOUND",
                "contract": contract,
                "contract_sha256": contract_sha,
            }), encoding="utf-8")
            prediction = eval_dir / "wmt24_predictions.jsonl"
            prediction.write_text("{}\n", encoding="utf-8")
            chemistry_summary = {
                "status": "PASS",
                "eval_contract_sha256": contract_sha,
                "slices": {"all": {"strict_canonical": {"accuracy": 0.2}}},
            }
            general_summary = {
                "status": "PASS",
                "eval_contract_sha256": contract_sha,
                "datasets": {"WMT24": {
                    "predictions_sha256": tb.sha256_file(prediction),
                }},
                "macro_bleu": 10.0,
                "macro_chrf": 20.0,
            }
            eval_manifest = {
                "status": "PASS",
                "arm": arm,
                "eval_contract": contract,
                "eval_contract_sha256": contract_sha,
                "frozen_inputs": {
                    "targeted": {
                        "status": "PASS", "actual": "same", "expected": "same",
                    },
                },
            }
            experiment = bridge.Experiment(
                run_id="matched", root=root, arm=arm, arm_dir=arm_dir,
                manifest_path=manifest_path, training_log=None, method="sft",
            )
            evidence, sources = bridge.validate_bound_evaluation_contract(
                experiment, train_manifest, eval_dir, eval_manifest,
                chemistry_summary, general_summary,
            )
            self.assertEqual(evidence["status"], "PASS")
            self.assertEqual(evidence["checkpoint_step"], 1032)
            self.assertIn(eval_dir / "eval_contract.json", sources)
            metric_rows = bridge.chemistry_records(
                chemistry_summary, step=1032,
                path=eval_dir / "chemistry_summary.json", eval_id="matched",
            ) + bridge.quality_records(
                general_summary, step=1032,
                path=eval_dir / "general_summary.json", eval_id="matched",
            )
            values = {row.tag: row.value for row in metric_rows}
            self.assertEqual(values["knowledge/chemistry_strict"], 0.2)
            self.assertEqual(values["quality/bleu"], 10.0)
            self.assertEqual(values["quality/chrf"], 20.0)
            bad_summary = dict(chemistry_summary)
            bad_summary["eval_contract_sha256"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "summary contract mismatch"):
                bridge.validate_bound_evaluation_contract(
                    experiment, train_manifest, eval_dir, eval_manifest,
                    bad_summary, general_summary,
                )

            prediction.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prediction SHA mismatch"):
                bridge.validate_bound_evaluation_contract(
                    experiment, train_manifest, eval_dir, eval_manifest,
                    chemistry_summary, general_summary,
                )
            prediction.write_text("{}\n", encoding="utf-8")
            (checkpoint / "hf" / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "durable checkpoint model (size|SHA) mismatch"):
                bridge.validate_bound_evaluation_contract(
                    experiment, train_manifest, eval_dir, eval_manifest,
                    chemistry_summary, general_summary,
                )

    def test_final_step_requires_consistent_durable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "C2" / "train"
            train.mkdir(parents=True)
            manifest_path = train / "manifest.json"
            manifest = {"arm": "C2", "training": {"total_updates": 10}}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (train / "resume_meta.json").write_text(
                json.dumps({"completed_updates": 10}), encoding="utf-8",
            )
            progress = train / "progress.json"
            progress.write_text(
                json.dumps({"status": "PASS", "completed_updates": 10}),
                encoding="utf-8",
            )
            experiment = bridge.Experiment(
                run_id="fixture", root=root, arm="C2",
                arm_dir=root / "C2", manifest_path=manifest_path,
                training_log=None, method="sft",
            )
            step, _, evidence = bridge._final_checkpoint_step(experiment, manifest)
            self.assertEqual(step, 10)
            self.assertEqual(set(evidence.values()), {10})
            progress.write_text(
                json.dumps({"status": "PASS", "completed_updates": 9}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "step disagreement"):
                bridge._final_checkpoint_step(experiment, manifest)

    def test_missing_run_directory_is_explicit_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "run directory missing"):
                bridge.discover_experiments("missing", Path(tmp) / "absent")

    def test_native_update_ledger_maps_matched_sft_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "updates.jsonl"
            path.write_text(json.dumps({
                "optimizer_update": 1,
                "group_token_mean_ce": 0.25,
                "grad_norm_before_clip": 2.5,
                "group_response_tokens": 480,
                "lr_after_step": 2e-5,
            }) + "\n", encoding="utf-8")
            rows = bridge.parse_native_update_metrics(path)
        values = {row.tag: row.value for row in rows}
        self.assertEqual(values["sft/ce_per_response_token"], 0.25)
        self.assertEqual(values["optim/grad_norm_preclip"], 2.5)
        self.assertEqual(values["train/response_tokens"], 480.0)
        self.assertEqual(values["optim/lr"], 2e-5)
        self.assertNotIn("sft/ce_microbatch_mean_recent50", values)

    def test_native_update_ledger_requires_contiguous_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "updates.jsonl"
            path.write_text("\n".join([
                json.dumps({"optimizer_update": 1, "group_token_mean_ce": 1.0}),
                json.dumps({"optimizer_update": 3, "group_token_mean_ce": 0.8}),
            ]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-contiguous"):
                bridge.parse_native_update_metrics(path)


if __name__ == "__main__":
    unittest.main()


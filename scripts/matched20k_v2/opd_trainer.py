#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
import json
import os
from pathlib import Path
import re

from verl.trainer.ppo.v1 import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

from scripts.matched20k_v2.shared_mt_validation import (
    compute_mt_metrics,
)


THINK_RE = re.compile(
    r"</?(?:think|analysis)>",
    flags=re.IGNORECASE,
)


@register_trainer("matched20k_sync")
class Matched20kPPOTrainerSync(PPOTrainerSync):
    """
    PPO/OPD trainer for the matched20k_v2 comparison.

    Training semantics are inherited from Verl unchanged.

    This subclass only adds:
      1. strict validation-contract checks;
      2. capture of Verl-native validation generations;
      3. shared MT BLEU/chrF metrics returned through Verl's logger.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        path = os.environ.get(
            "MATCHED20K_VALIDATION_JSONL"
        )

        if not path:
            raise RuntimeError(
                "MATCHED20K_VALIDATION_JSONL is required"
            )

        self._matched_validation_path = Path(path)

        if not self._matched_validation_path.is_file():
            raise FileNotFoundError(
                self._matched_validation_path
            )

        self._matched_rows = None
        self._captured_val_inputs = None
        self._captured_val_outputs = None

        self._validate_matched_config_contract()

    def fit(self, agent_loop_manager):
        """
        Attach an external checkpoint's optimizer-step label to
        Verl's native validation-only path.

        This is metadata-only glue. It is forbidden for training
        runs and does not replace Verl validation, generation,
        logging, or checkpoint logic.
        """
        raw_step = self.config.trainer.get(
            "validation_step_override",
            None,
        )

        if raw_step is not None:
            if not self.config.trainer.get(
                "val_only",
                False,
            ):
                raise RuntimeError(
                    "trainer.validation_step_override "
                    "requires trainer.val_only=true"
                )

            if not self.config.trainer.get(
                "val_before_train",
                True,
            ):
                raise RuntimeError(
                    "trainer.validation_step_override "
                    "requires trainer.val_before_train=true"
                )

            if self.config.trainer.get(
                "resume_mode",
                "disable",
            ) != "disable":
                raise RuntimeError(
                    "validation-only external checkpoint "
                    "must not use trainer resume"
                )

            if isinstance(raw_step, bool):
                raise TypeError(
                    "validation_step_override must be int"
                )

            step = int(raw_step)

            if step < 0 or step != raw_step:
                raise ValueError(
                    "validation_step_override must be "
                    "a non-negative integer"
                )

            if self.global_steps != 0:
                raise RuntimeError(
                    "validation_step_override expects "
                    "fresh Verl evaluator global_steps=0, "
                    f"got {self.global_steps}"
                )

            self.global_steps = step

        return super().fit(
            agent_loop_manager
        )

    def _validate_matched_config_contract(self):
        cfg = self.config

        if bool(cfg.data.get("validation_shuffle", True)):
            raise RuntimeError(
                "matched20k requires data.validation_shuffle=False"
            )

        thinking = (
            cfg.data
            .get("apply_chat_template_kwargs", {})
            .get("enable_thinking", None)
        )

        if thinking is not False:
            raise RuntimeError(
                "matched20k requires "
                "data.apply_chat_template_kwargs."
                "enable_thinking=False"
            )

        val_kwargs = cfg.actor_rollout_ref.rollout.val_kwargs

        if bool(val_kwargs.get("do_sample", True)):
            raise RuntimeError(
                "matched20k validation requires do_sample=False"
            )

        n = int(val_kwargs.get("n", 1))

        if n != 1:
            raise RuntimeError(
                f"matched20k validation requires n=1, got {n}"
            )

    def _load_matched_rows(self):
        if self._matched_rows is not None:
            return self._matched_rows

        rows = []

        with self._matched_validation_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"invalid validation JSON "
                        f"at line {line_no}"
                    ) from exc

                rows.append(row)

        if len(rows) != 3231:
            raise RuntimeError(
                f"validation rows={len(rows)}, expected=3231"
            )

        self._matched_rows = rows
        return rows

    def _render_expected_input(self, messages):
        kwargs = dict(
            self.config.data.get(
                "apply_chat_template_kwargs",
                {}
            )
        )

        # Hard contract even if config changes upstream.
        kwargs["enable_thinking"] = False

        kwargs.pop("tokenize", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("return_tensors", None)

        token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            **kwargs,
        )

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def _align_outputs_to_frozen_rows(
        self,
        inputs,
        outputs,
    ):
        rows = self._load_matched_rows()

        if len(inputs) != len(outputs):
            raise RuntimeError(
                "captured validation input/output "
                f"length mismatch: {len(inputs)} vs {len(outputs)}"
            )

        if len(outputs) != len(rows):
            raise RuntimeError(
                "captured validation output count "
                f"{len(outputs)} != frozen rows {len(rows)}"
            )

        # Replay-buffer/agent-loop completion order need not equal
        # parquet order. Align by the exact rendered prompt instead
        # of assuming asynchronous generation order.
        expected_by_input = defaultdict(deque)

        for row in rows:
            rendered = self._render_expected_input(
                row["messages"]
            )
            expected_by_input[rendered].append(row)

        aligned_rows = []
        aligned_outputs = []

        for i, (input_text, output_text) in enumerate(
            zip(inputs, outputs)
        ):
            queue = expected_by_input.get(input_text)

            if not queue:
                raise RuntimeError(
                    "validation prompt could not be matched "
                    f"at generated position {i}: "
                    f"{input_text[:240]!r}"
                )

            row = queue.popleft()

            aligned_rows.append(row)
            aligned_outputs.append(output_text)

        leftovers = sum(
            len(q)
            for q in expected_by_input.values()
        )

        if leftovers != 0:
            raise RuntimeError(
                f"validation alignment left {leftovers} "
                "unconsumed frozen rows"
            )

        return aligned_rows, aligned_outputs

    def _maybe_log_val_generations(
        self,
        inputs,
        outputs,
        scores,
    ):
        # Capture the exact generations produced by Verl's native
        # validation path before the normal logger handles them.
        self._captured_val_inputs = list(inputs)
        self._captured_val_outputs = list(outputs)

        return super()._maybe_log_val_generations(
            inputs=inputs,
            outputs=outputs,
            scores=scores,
        )

    def _validate(self) -> dict[str, float]:
        self._captured_val_inputs = None
        self._captured_val_outputs = None

        metrics = super()._validate()

        if self._captured_val_inputs is None:
            raise RuntimeError(
                "Verl validation inputs were not captured"
            )

        if self._captured_val_outputs is None:
            raise RuntimeError(
                "Verl validation outputs were not captured"
            )

        rows, outputs = self._align_outputs_to_frozen_rows(
            self._captured_val_inputs,
            self._captured_val_outputs,
        )

        thinking_rows = [
            i
            for i, text in enumerate(outputs)
            if THINK_RE.search(text or "")
        ]

        if thinking_rows:
            raise RuntimeError(
                "non-thinking validation contract violated; "
                f"reasoning markers found in {len(thinking_rows)} "
                f"outputs, first={thinking_rows[:20]}"
            )

        mt_metrics = compute_mt_metrics(
            data_sources=[
                row["eval_group"]
                for row in rows
            ],
            references=[
                row["reference"]
                for row in rows
            ],
            predictions=outputs,
            strict_counts=True,
        )

        # These are returned through the ordinary Verl validation
        # result, so fit() writes them via the native Tracking logger.
        metrics.update(mt_metrics)

        return metrics


@register_trainer("matched20k_persistent_replay")
class Matched20kPersistentReplayTrainer(
    Matched20kPPOTrainerSync
):
    """
    Post-hoc matched20k checkpoint validation with one persistent
    Verl/Ray/rollout lifecycle.

    All generation, checkpoint loading, weight synchronization,
    validation, and logging remain Verl-native.

    Project-side responsibility is limited to:
      1. enumerating frozen checkpoint steps;
      2. asking Verl to load each actor checkpoint;
      3. invoking the existing native validation path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        root = os.environ.get(
            "MATCHED20K_REPLAY_CHECKPOINT_ROOT"
        )
        raw_steps = os.environ.get(
            "MATCHED20K_REPLAY_STEPS"
        )

        if not root:
            raise RuntimeError(
                "MATCHED20K_REPLAY_CHECKPOINT_ROOT is required"
            )

        if not raw_steps:
            raise RuntimeError(
                "MATCHED20K_REPLAY_STEPS is required"
            )

        self._replay_checkpoint_root = Path(root)

        if not self._replay_checkpoint_root.is_dir():
            raise FileNotFoundError(
                self._replay_checkpoint_root
            )

        try:
            steps = [
                int(x.strip())
                for x in raw_steps.split(",")
                if x.strip()
            ]
        except ValueError as exc:
            raise RuntimeError(
                "MATCHED20K_REPLAY_STEPS must be "
                "comma-separated integers"
            ) from exc

        if not steps:
            raise RuntimeError(
                "persistent replay step list is empty"
            )

        if any(step <= 0 for step in steps):
            raise RuntimeError(
                "persistent replay steps must be > 0; "
                "step 0 is evaluated from the initial model"
            )

        if steps != sorted(set(steps)):
            raise RuntimeError(
                "persistent replay steps must be strictly "
                "increasing and unique"
            )

        self._replay_steps = steps
        self._defer_dump_executor_shutdown = False

        self._validate_persistent_replay_contract()

    def _validate_persistent_replay_contract(self):
        trainer = self.config.trainer

        if not trainer.get("val_only", False):
            raise RuntimeError(
                "persistent replay requires trainer.val_only=true"
            )

        if not trainer.get("val_before_train", True):
            raise RuntimeError(
                "persistent replay requires "
                "trainer.val_before_train=true"
            )

        if trainer.get(
            "resume_mode",
            "disable",
        ) != "disable":
            raise RuntimeError(
                "persistent replay requires "
                "trainer.resume_mode=disable"
            )

        raw_step = trainer.get(
            "validation_step_override",
            0,
        )

        if int(raw_step) != 0:
            raise RuntimeError(
                "persistent replay initial validation "
                "must be labeled step 0"
            )

        if bool(self.config.distillation.enabled):
            raise RuntimeError(
                "persistent validation must run with "
                "distillation.enabled=false"
            )

        for step in self._replay_steps:
            actor_dir = (
                self._replay_checkpoint_root
                / f"global_step_{step}"
                / "actor"
            )

            if not actor_dir.is_dir():
                raise FileNotFoundError(actor_dir)

            model_shards = list(
                actor_dir.glob(
                    "model_world_size_*_rank_*.pt"
                )
            )

            if not model_shards:
                raise RuntimeError(
                    f"no native model shards in {actor_dir}"
                )

    def _shutdown_dump_executor(self):
        """
        Verl's val-only fit normally closes its dump executor
        immediately after the first validation.

        Persistent replay deliberately keeps it alive until all
        checkpoint validations have completed.
        """
        if self._defer_dump_executor_shutdown:
            return

        return super()._shutdown_dump_executor()

    def fit(self, agent_loop_manager):
        self._defer_dump_executor_shutdown = True

        try:
            # Native Verl val-only path:
            #   * initializes logger / agent loop / replay state;
            #   * validates the initial Student at step 0;
            #   * returns without any optimizer update.
            super().fit(agent_loop_manager)

            for step in self._replay_steps:
                actor_dir = (
                    self._replay_checkpoint_root
                    / f"global_step_{step}"
                    / "actor"
                )

                print(
                    "MATCHED20K_PERSISTENT_REPLAY_LOAD "
                    f"step={step} "
                    f"actor_dir={actor_dir}",
                    flush=True,
                )

                # Rollout currently contains the previous checkpoint.
                # Release its weights/KV state before replacing the
                # colocated actor weights.
                self.checkpoint_manager.sleep_replicas()

                # Native Verl FSDP checkpoint load.
                self.actor_rollout_wg.load_checkpoint(
                    local_path=str(actor_dir),
                    del_local_after_load=False,
                )

                self.global_steps = step

                # Native Verl actor -> rollout weight synchronization.
                self.checkpoint_manager.update_weights(
                    self.global_steps
                )

                self.on_validate_begin()
                metrics = self._validate()
                self.on_validate_end()

                if not metrics:
                    raise RuntimeError(
                        f"empty validation metrics at step {step}"
                    )

                # Same native Tracking logger and step semantics used
                # by ordinary Verl validation.
                self.logger.log(
                    data=metrics,
                    step=self.global_steps,
                )

                print(
                    "MATCHED20K_PERSISTENT_REPLAY_STEP "
                    f"{step}=PASS",
                    flush=True,
                )

        finally:
            self._defer_dump_executor_shutdown = False
            super()._shutdown_dump_executor()

# Pre-M1 formal code synchronization (2026-10-19)

This commit preserves formal-research source code and CPU-contract tooling. **It is not an independently runnable training environment.**

- Original local project HEAD: `958e8e72f3d86720aa0d5e78651fa0a561f71edc`
- GitHub review base: `b17af747ae905fd6381d593266fec3810f991082`
- Local post-base commits: `4026dee7d4c526d134f8a6cbd8c2b220778e9c00, 24919e004aaa6cd6697d1f98fd00c7624e3cc1fd, 667db120e94166bab451e893aded2394bfdc2214, 958e8e72f3d86720aa0d5e78651fa0a561f71edc`
- History integration method: `ISOLATED_SNAPSHOT_PROVENANCE_ONLY`
- Reviewed source counts: `{'chem_eval': 2, 'chem_opd': 5, 'clean_sft': 23, 'mixed_gkd': 27, 'tensorboard': 7, 'tests': 1}`

Excluded from this public GitHub sync: M1–M6 mechanism probes and datasets, unreviewed 2026-10-10 full audit, checkpoint/model files, runtime logs, TensorBoard events, generated graphics, and raw Verl vendor modifications.

External, version-pinned but **not vendored** dependencies include native Verl v0.9.0 with local adaptations, the server-side Chemistry Stage1 data/setup scripts, frozen Chemistry1000 evaluator, Teacher targets, and model weights. Review the per-experiment READMEs and do not claim standalone reproduction from this GitHub branch alone.

This GitHub branch does not authorize starting formal training or modifying existing experiment outputs.

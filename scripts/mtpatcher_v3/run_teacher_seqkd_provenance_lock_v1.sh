#!/usr/bin/env bash
set -Eeuo pipefail

source /workspace/mtpatcher/project_env.sh

EXP="mtpatcher_v3_full6565_20260823"
D="$DATA_ROOT/$EXP"

OUT="$D/teacher_seqkd_provenance_lock_v1"

BROAD="$D/paperfaith_paper20k_v2/demo_pool.jsonl"
BROAD_JOBS="$D/paperfaith_paper20k_v2/feedback_jobs.jsonl"

OLD="/workspace/mtpatcher/data/mtpatcher_v3_full6565_20260823/rq0_seqkd_scaling50k_v1/seqkd_newscrawl20000_qwen3_8b_v1.jsonl"

MODEL="$MODEL_ROOT/Qwen3-8B"

PASS="$OUT/TEACHER_SEQKD_PROVENANCE_LOCK_V1.PASS"
FAIL="$OUT/TEACHER_SEQKD_PROVENANCE_LOCK_V1.FAIL"

REPORT_JSON="$OUT/teacher_seqkd_provenance_report_v1.json"
REPORT_TXT="$OUT/teacher_seqkd_provenance_report_v1.txt"

mkdir -p "$OUT"

rm -f "$PASS" "$FAIL"

trap '
rc=$?
echo
echo "======================================================================"
echo "TEACHER SEQKD PROVENANCE LOCK V1 FAILED"
echo "return_code=$rc"
date
echo "======================================================================"
touch "'"$FAIL"'"
' ERR

echo "======================================================================"
echo "TEACHER SEQKD PROVENANCE LOCK V1"
date
echo "======================================================================"

echo
echo "D=$D"
echo "BROAD=$BROAD"
echo "BROAD_JOBS=$BROAD_JOBS"
echo "OLD=$OLD"
echo "MODEL=$MODEL"
echo


###############################################################################
# STAGE 1/4
# Static asset + source-overlap + metadata audit
###############################################################################

echo "======================================================================"
echo "STAGE 1/4: ASSET / SOURCE / METADATA AUDIT"
echo "======================================================================"

python - \
    "$D" \
    "$ROOT" \
    "$LOG_ROOT" \
    "$MODEL" \
    "$BROAD" \
    "$BROAD_JOBS" \
    "$OLD" \
    "$OUT" <<'PY'

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(sys.argv[1])
ROOT = Path(sys.argv[2])
LOG_ROOT = Path(sys.argv[3])
MODEL = Path(sys.argv[4])
BROAD = Path(sys.argv[5])
BROAD_JOBS = Path(sys.argv[6])
OLD = Path(sys.argv[7])
OUT = Path(sys.argv[8])

REPORT_JSON = OUT / "teacher_seqkd_provenance_report_v1.json"
REPORT_TXT = OUT / "teacher_seqkd_provenance_report_v1.txt"


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_jsonl(path):
    rows = []

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))

            except Exception as e:
                raise RuntimeError(
                    f"JSON parse failed "
                    f"path={path} "
                    f"line={line_no}: {e}"
                )

    return rows


def canonical(x):
    return json.dumps(
        x,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def get_source(row):
    x = row.get("source")

    if isinstance(x, str):
        return x

    return None


def get_id(row):

    for key in (
        "index",
        "demo_id",
        "id",
    ):
        if row.get(key) is not None:
            return str(row[key])

    return None


def get_possible_target(row):

    preferred = [
        "teacher_translation",
        "translation",
        "target",
        "response",
        "output",
        "student_translation",
        "reference",
    ]

    found = {}

    for key in preferred:

        x = row.get(key)

        if isinstance(x, str) and x.strip():
            found[key] = x.strip()

    messages = row.get("messages")

    if isinstance(messages, list):

        assistants = []

        for m in messages:

            if not isinstance(m, dict):
                continue

            role = m.get("role")
            content = m.get("content")

            if (
                role == "assistant"
                and isinstance(content, str)
                and content.strip()
            ):
                assistants.append(content.strip())

        if assistants:
            found["messages:last_assistant"] = assistants[-1]

    return found


def distinct_json_values(rows, key, cap=20):

    c = Counter()

    for row in rows:

        if key not in row:
            continue

        try:
            v = canonical(row[key])
        except Exception:
            v = repr(row[key])

        c[v] += 1

    return [
        {
            "value": k,
            "count": v,
        }
        for k, v in c.most_common(cap)
    ]


def summarize_asset(path, load_all=True):

    result = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return result

    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256(path)

    rows = read_jsonl(path)

    result["rows"] = len(rows)

    if not rows:
        result["keys"] = []
        return result

    key_union = set()

    for x in rows[:500]:
        key_union.update(x.keys())

    result["keys"] = sorted(key_union)

    sources = [
        get_source(x)
        for x in rows
    ]

    sources_nonnull = [
        x for x in sources
        if isinstance(x, str)
    ]

    result["source_nonnull"] = len(sources_nonnull)
    result["source_unique"] = len(set(sources_nonnull))
    result["source_duplicate_rows"] = (
        len(sources_nonnull)
        - len(set(sources_nonnull))
    )

    ids = [
        get_id(x)
        for x in rows
    ]

    ids_nonnull = [
        x for x in ids
        if x is not None
    ]

    result["id_nonnull"] = len(ids_nonnull)
    result["id_unique"] = len(set(ids_nonnull))

    target_counts = Counter()

    for x in rows:

        for key in get_possible_target(x):
            target_counts[key] += 1

    result["possible_target_fields"] = dict(
        target_counts
    )

    for key in (
        "generation_config",
        "teacher_model_path",
        "model_path",
        "model",
        "evaluation_method",
        "generation_method",
    ):
        vals = distinct_json_values(rows, key)

        if vals:
            result[f"{key}_values"] = vals

    return result


if not BROAD.exists():
    raise RuntimeError(
        f"required Broad20k pool missing: {BROAD}"
    )

broad_rows = read_jsonl(BROAD)

if len(broad_rows) != 20000:
    raise RuntimeError(
        f"Broad20k expected 20000, "
        f"got {len(broad_rows)}"
    )

broad_sources = [
    get_source(x)
    for x in broad_rows
]

if any(x is None for x in broad_sources):
    raise RuntimeError(
        "Broad20k contains rows without source"
    )

broad_source_counter = Counter(broad_sources)

old_rows = (
    read_jsonl(OLD)
    if OLD.exists()
    else []
)

old_sources = [
    get_source(x)
    for x in old_rows
]

old_sources_nonnull = [
    x
    for x in old_sources
    if isinstance(x, str)
]

old_source_counter = Counter(
    old_sources_nonnull
)

broad_set = set(broad_sources)
old_set = set(old_sources_nonnull)

overlap = broad_set & old_set


###############################################################################
# Find all plausible SeqKD JSONL assets under experiment directory.
###############################################################################

seqkd_assets = []

for p in sorted(D.rglob("*.jsonl")):

    if "seqkd" not in p.name.lower():
        continue

    try:
        info = summarize_asset(p)

    except Exception as e:
        info = {
            "path": str(p),
            "exists": True,
            "inspection_error": repr(e),
        }

    seqkd_assets.append(info)


###############################################################################
# Model / tokenizer provenance hashes.
###############################################################################

model_files = {}

for name in (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
):

    p = MODEL / name

    if p.exists():

        model_files[name] = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
        }


###############################################################################
# Search repository scripts for SeqKD / Teacher generators.
###############################################################################

script_candidates = []

extensions = {
    ".py",
    ".sh",
}

patterns = [
    "seqkd",
    "teacher",
    "apply_chat_template",
    "enable_thinking",
    "do_sample",
    "max_new_tokens",
    "Qwen3-8B",
    "Translate",
    "translation",
]

for p in sorted((ROOT / "scripts").rglob("*")):

    if not p.is_file():
        continue

    if p.suffix not in extensions:
        continue

    try:
        if p.stat().st_size > 2_000_000:
            continue

        text = p.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception:
        continue

    low = text.lower()

    if (
        "seqkd" not in low
        and "teacher" not in low
    ):
        continue

    hits = []

    for line_no, line in enumerate(
        text.splitlines(),
        1,
    ):

        if any(
            pat.lower() in line.lower()
            for pat in patterns
        ):
            hits.append({
                "line": line_no,
                "text": line[:500],
            })

    if hits:

        script_candidates.append({
            "path": str(p),
            "sha256": sha256(p),
            "hits": hits[:120],
        })


###############################################################################
# Search exact references to the old Broad SeqKD asset.
###############################################################################

literal_names = [
    OLD.name,
    "seqkd_newscrawl20000_qwen3_8b_v1",
]

reference_matches = []

search_roots = [
    ROOT / "scripts",
    LOG_ROOT / "mtpatcher_v3_full6565_20260823",
]

allowed_suffix = {
    ".py",
    ".sh",
    ".log",
    ".txt",
    ".json",
    ".md",
}

for root in search_roots:

    if not root.exists():
        continue

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        if p.suffix not in allowed_suffix:
            continue

        try:
            if p.stat().st_size > 20_000_000:
                continue

            text = p.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except Exception:
            continue

        lines = text.splitlines()

        for i, line in enumerate(lines):

            if not any(
                literal in line
                for literal in literal_names
            ):
                continue

            lo = max(0, i - 4)
            hi = min(len(lines), i + 5)

            reference_matches.append({
                "path": str(p),
                "line": i + 1,
                "context": [
                    {
                        "line": j + 1,
                        "text": lines[j][:700],
                    }
                    for j in range(lo, hi)
                ],
            })

            if len(reference_matches) >= 100:
                break

        if len(reference_matches) >= 100:
            break

    if len(reference_matches) >= 100:
        break


###############################################################################
# Summaries
###############################################################################

broad_summary = summarize_asset(BROAD)

jobs_summary = summarize_asset(
    BROAD_JOBS
)

old_summary = summarize_asset(
    OLD
)

report = {
    "protocol":
        "TEACHER_SEQKD_PROVENANCE_LOCK_V1",

    "scientific_role":
        "Engineering provenance gate before exact Broad20k Teacher SeqKD generation",

    "broad20k": broad_summary,

    "broad20k_feedback_jobs":
        jobs_summary,

    "old_seqkd20k":
        old_summary,

    "exact_source_overlap": {
        "broad_unique_sources":
            len(broad_set),

        "old_unique_sources":
            len(old_set),

        "intersection_unique_sources":
            len(overlap),

        "broad_duplicate_source_rows":
            sum(
                v - 1
                for v in broad_source_counter.values()
                if v > 1
            ),

        "old_duplicate_source_rows":
            sum(
                v - 1
                for v in old_source_counter.values()
                if v > 1
            ),

        "broad_sources_not_in_old":
            len(broad_set - old_set),

        "old_sources_not_in_broad":
            len(old_set - broad_set),
    },

    "model_provenance": {
        "model_path": str(MODEL),
        "files": model_files,
    },

    "seqkd_assets_found":
        seqkd_assets,

    "candidate_generation_scripts":
        script_candidates,

    "old_asset_creation_references":
        reference_matches,

    "reuse_gate": {
        "status": "REVIEW_REQUIRED",

        "must_prove": [
            "same exact Qwen3-8B checkpoint/model artifact",
            "same translation instruction / prompt",
            "same Qwen chat-template rendering",
            "same enable_thinking setting",
            "same decoding strategy and sampling parameters",
            "same max_new_tokens / stopping behavior",
            "same source field semantics",
            "deterministic/replay compatibility sufficient for artifact reuse",
        ],

        "important_warning": (
            "do_sample=False alone is not treated as proof "
            "of bitwise artifact reproducibility on the current "
            "NPU/backend stack."
        ),

        "decision_rule": (
            "Reuse overlap only after provenance review. "
            "If any material generation-spec field is unresolved "
            "or mismatched, regenerate all 20000 Teacher targets."
        ),
    },
}

REPORT_JSON.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


###############################################################################
# Human-readable report.
###############################################################################

lines = []

def emit(x=""):
    lines.append(str(x))


emit("=" * 78)
emit("TEACHER SEQKD PROVENANCE LOCK V1")
emit("=" * 78)

emit()
emit("===== BROAD20K =====")
emit(f"path = {BROAD}")
emit(f"rows = {broad_summary.get('rows')}")
emit(
    "unique_sources = "
    f"{broad_summary.get('source_unique')}"
)
emit(
    "duplicate_source_rows = "
    f"{broad_summary.get('source_duplicate_rows')}"
)
emit(
    "keys = "
    f"{broad_summary.get('keys')}"
)

emit()
emit("===== OLD SEQKD20K =====")
emit(f"path = {OLD}")
emit(f"exists = {OLD.exists()}")

if OLD.exists():

    emit(f"rows = {old_summary.get('rows')}")
    emit(
        "unique_sources = "
        f"{old_summary.get('source_unique')}"
    )
    emit(
        "keys = "
        f"{old_summary.get('keys')}"
    )
    emit(
        "possible_target_fields = "
        f"{old_summary.get('possible_target_fields')}"
    )

    for key in (
        "generation_config_values",
        "teacher_model_path_values",
        "model_path_values",
        "model_values",
        "evaluation_method_values",
    ):

        if key in old_summary:
            emit(
                f"{key} = "
                f"{old_summary[key]}"
            )

emit()
emit("===== EXACT SOURCE OVERLAP =====")
emit(
    "intersection_unique_sources = "
    f"{len(overlap)}"
)
emit(
    "broad_missing_from_old = "
    f"{len(broad_set - old_set)}"
)
emit(
    "old_not_in_broad = "
    f"{len(old_set - broad_set)}"
)

emit()
emit("===== MODEL / TOKENIZER HASHES =====")

for name, info in model_files.items():

    emit(
        f"{name}: "
        f"sha256={info['sha256']} "
        f"size={info['size_bytes']}"
    )

emit()
emit("===== OLD ASSET CREATION REFERENCES =====")
emit(
    f"matches = {len(reference_matches)}"
)

for m in reference_matches[:30]:

    emit()
    emit(
        f"{m['path']}:{m['line']}"
    )

    for c in m["context"]:

        emit(
            f"  {c['line']}: "
            f"{c['text']}"
        )

emit()
emit("===== CANDIDATE GENERATION SCRIPTS =====")
emit(
    f"count = {len(script_candidates)}"
)

for s in script_candidates[:40]:

    emit()
    emit(
        f"SCRIPT {s['path']}"
    )
    emit(
        f"SHA256 {s['sha256']}"
    )

    for h in s["hits"][:50]:

        emit(
            f"  {h['line']}: "
            f"{h['text']}"
        )

emit()
emit("===== SEQKD JSONL ASSETS FOUND =====")
emit(
    f"count = {len(seqkd_assets)}"
)

for a in seqkd_assets:

    emit()
    emit(f"ASSET {a.get('path')}")
    emit(f"  rows={a.get('rows')}")
    emit(f"  sha256={a.get('sha256')}")
    emit(f"  keys={a.get('keys')}")
    emit(
        "  possible_target_fields="
        f"{a.get('possible_target_fields')}"
    )

    for key in (
        "generation_config_values",
        "teacher_model_path_values",
        "model_path_values",
        "model_values",
        "evaluation_method_values",
    ):

        if key in a:
            emit(
                f"  {key}={a[key]}"
            )

emit()
emit("===== REUSE GATE =====")
emit("STATUS = REVIEW_REQUIRED")
emit(
    "No generation is authorized by this script."
)
emit(
    "If any material Teacher generation specification "
    "is unresolved or mismatched, regenerate full20k."
)

REPORT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print()
print("BROAD_ROWS =", len(broad_rows))
print(
    "BROAD_UNIQUE_SOURCES =",
    len(broad_set),
)

print(
    "OLD_EXISTS =",
    OLD.exists(),
)

if OLD.exists():
    print(
        "OLD_ROWS =",
        len(old_rows),
    )

print(
    "EXACT_SOURCE_OVERLAP =",
    len(overlap),
)

print(
    "BROAD_MISSING_FROM_OLD =",
    len(broad_set - old_set),
)

print(
    "OLD_NOT_IN_BROAD =",
    len(old_set - broad_set),
)

print(
    "OLD_ASSET_CREATION_REFERENCE_MATCHES =",
    len(reference_matches),
)

print(
    "CANDIDATE_GENERATION_SCRIPTS =",
    len(script_candidates),
)

print(
    "SEQKD_ASSETS_FOUND =",
    len(seqkd_assets),
)

print()
print("REPORT_JSON =", REPORT_JSON)
print("REPORT_TXT =", REPORT_TXT)
print()
print("TEACHER_SEQKD_PROVENANCE_AUDIT_PASS")
PY


###############################################################################
# STAGE 2/4
# Grep specifically for generation semantics in likely scripts.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 2/4: TARGETED SCRIPT EVIDENCE"
echo "======================================================================"

{
    echo "===== FILES NAMED LIKE SEQKD GENERATORS ====="

    find "$ROOT/scripts" \
        -type f \
        \( -iname '*seqkd*.py' \
        -o -iname '*seqkd*.sh' \
        -o -iname '*teacher*.py' \
        -o -iname '*teacher*.sh' \) \
        -print \
        2>/dev/null \
        | sort

    echo
    echo "===== RELEVANT GENERATION LINES ====="

    grep -RInE \
        'apply_chat_template|enable_thinking|do_sample|max_new_tokens|temperature|top_p|Qwen3-8B|translate|translation|seqkd' \
        "$ROOT/scripts" \
        2>/dev/null \
        | grep -Ei \
        'seqkd|teacher' \
        | head -n 600 \
        || true

    echo
    echo "===== OLD ASSET LITERAL REFERENCES ====="

    grep -RInF \
        'seqkd_newscrawl20000_qwen3_8b_v1' \
        "$ROOT/scripts" \
        "$LOG_ROOT/$EXP" \
        2>/dev/null \
        | head -n 200 \
        || true

} > "$OUT/targeted_generation_evidence_v1.txt"

cat "$OUT/targeted_generation_evidence_v1.txt"


###############################################################################
# STAGE 3/4
# Freeze hashes.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 3/4: HASH / CARDINALITY FREEZE"
echo "======================================================================"

{
    echo "DATE=$(date -Iseconds)"

    echo
    echo "===== PRIMARY FILES ====="

    for F in \
        "$BROAD" \
        "$BROAD_JOBS" \
        "$OLD" \
        "$REPORT_JSON" \
        "$REPORT_TXT" \
        "$OUT/targeted_generation_evidence_v1.txt"
    do

        if [ -f "$F" ]; then

            printf "%s  " "$(sha256sum "$F" | awk '{print $1}')"
            echo "$F"

        else

            echo "MISSING  $F"

        fi

    done

    echo
    echo "===== MODEL FILES ====="

    for F in \
        "$MODEL/config.json" \
        "$MODEL/generation_config.json" \
        "$MODEL/tokenizer_config.json" \
        "$MODEL/tokenizer.json" \
        "$MODEL/special_tokens_map.json"
    do

        if [ -f "$F" ]; then

            printf "%s  " "$(sha256sum "$F" | awk '{print $1}')"
            echo "$F"

        fi

    done

} > "$OUT/provenance_hash_manifest_v1.txt"

cat "$OUT/provenance_hash_manifest_v1.txt"


###############################################################################
# STAGE 4/4
# Finalize. This PASS means AUDIT COMPLETED, not REUSE APPROVED.
###############################################################################

echo
echo "======================================================================"
echo "STAGE 4/4: FINALIZE"
echo "======================================================================"

touch "$PASS"
rm -f "$FAIL"

echo
echo "======================================================================"
echo "TEACHER SEQKD PROVENANCE LOCK V1 PASS"
echo
echo "IMPORTANT:"
echo "PASS means provenance audit completed."
echo "It does NOT mean old7974 reuse is approved yet."
echo
echo "REPORT_JSON=$REPORT_JSON"
echo "REPORT_TXT=$REPORT_TXT"
echo "TARGETED=$OUT/targeted_generation_evidence_v1.txt"
echo "HASH_MANIFEST=$OUT/provenance_hash_manifest_v1.txt"
echo
date
echo "======================================================================"


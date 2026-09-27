#!/usr/bin/env python3
from __future__ import annotations

from difflib import SequenceMatcher

from scripts.pe_pds_v1.eaec_core_v1 import (
    words_with_offsets,
)


STOP = set(
    """
    the a an and or of to in on at for from with by as
    is are was were be been being this that these those
    it its their his her them they he she we you i but
    if then than so such into over under before after
    during through about out up down no not only even
    more most very also still when where who which while
    will would can could should may might do does did
    have has had said says stated according report reports
    people china chinese year years
    """.split()
)


def _tokens(text: str) -> list[str]:
    return [
        str(x[0]).lower()
        for x in words_with_offsets(text)
    ]


def historical_unique_old_core(
    old_span: str,
    correction: str,
) -> set[str]:
    """
    Conservative old-side edit core.

    A token is retained only if:
      1. it belongs to a non-equal old↔correction opcode;
      2. it does not appear anywhere in correction;
      3. it is not a generic stopword;
      4. it has lexical content.

    Thus the token has explicit provenance from something
    MT-PATCHER's correction removed/replaced.
    """

    old = _tokens(old_span)
    corr = _tokens(correction)

    sm = SequenceMatcher(
        None,
        old,
        corr,
        autojunk=False,
    )

    changed_old: list[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        changed_old.extend(
            old[i1:i2]
        )

    corr_set = set(corr)

    return {
        tok
        for tok in changed_old
        if (
            tok not in corr_set
            and tok not in STOP
            and len(tok) >= 3
        )
    }


def strict_surviving_old_core(
    *,
    old_span: str,
    correction: str,
    current_region: str,
    min_distinct_terms: int = 2,
) -> dict:
    """
    Diagnostic-only v2 gate.

    It proposes an active historical edit core only when
    >= min_distinct_terms old-side MT-PATCHER edit terms
    still survive lexically inside the already recovered
    coarse current-trajectory region.

    No surrounding region is automatically selected.
    """

    historical_core = (
        historical_unique_old_core(
            old_span,
            correction,
        )
    )

    current_tokens = _tokens(
        current_region
    )

    survivors = sorted({
        tok
        for tok in current_tokens
        if tok in historical_core
    })

    return {
        "candidate":
            len(survivors)
            >= min_distinct_terms,

        "historical_core_terms":
            sorted(historical_core),

        "surviving_core_terms":
            survivors,

        "surviving_core_distinct":
            len(survivors),

        "min_distinct_terms":
            int(min_distinct_terms),
    }

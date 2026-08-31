#!/usr/bin/env python
"""One-off: report RSSM.dynamic's tensor dtypes, to locate the Byte tensor behind
`RuntimeError: Autograd not support dtype: Byte`.

Read-only -- coerces nothing, so the run still fails where it failed before. Prints the first call's
dtypes as a baseline, then any call whose dtypes differ from it. Delete once the cause is known.

Same overrides as resume.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "sheeprl"))

import resume  # noqa: E402


def patch_probe() -> None:
    from sheeprl.algos.dreamer_v3.agent import RSSM

    original = RSSM.dynamic
    baseline = None
    calls = 0

    def dynamic(self, posterior, recurrent_state, action, embedded_obs, is_first):
        nonlocal baseline, calls
        seen = {
            "recurrent_state": recurrent_state.dtype,
            "is_first": is_first.dtype,
            "posterior": posterior.dtype,
            "action": action.dtype,
            "embedded_obs": embedded_obs.dtype,
        }
        calls += 1
        if baseline is None:
            baseline = seen
            print(f"[probe] baseline at call 1: {seen}", flush=True)
        elif seen != baseline:
            print(f"[probe] DTYPE CHANGED at call {calls}: {seen}", flush=True)
            baseline = seen
        return original(self, posterior, recurrent_state, action, embedded_obs, is_first)

    RSSM.dynamic = dynamic


if __name__ == "__main__":
    resume.patch_torch_load()
    patch_probe()
    raise SystemExit(resume.main())

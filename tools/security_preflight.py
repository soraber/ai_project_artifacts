"""Fail closed for historical GPU recipes with unresolved dependency advisories."""


def require_gpu_clearance():
    raise RuntimeError(
        'New GPU execution is blocked by the 2026-11-17 dependency audit. '
        'Historical training/inference snapshots contain known advisories. '
        'See SECURITY_UPGRADE.md; migrate, audit, and smoke-test a replacement '
        'before enabling a new run. CPU regression checks remain available.'
    )

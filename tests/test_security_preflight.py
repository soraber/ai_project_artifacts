import importlib.util
from pathlib import Path

import pytest


def test_legacy_gpu_recipes_fail_closed():
    path = Path(__file__).parents[1] / 'tools/security_preflight.py'
    spec = importlib.util.spec_from_file_location('security_preflight', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match='blocked'):
        module.require_gpu_clearance()

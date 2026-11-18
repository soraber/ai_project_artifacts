"""Validate the installed training distributions against the reviewed lock."""
import re
import sys
from importlib import metadata
from pathlib import Path


def validate(lock_path):
    if sys.prefix == sys.base_prefix or not (sys.version_info[:2] == (3, 11) and sys.version_info[:3] >= (3, 11, 5)):
        raise RuntimeError('Select an isolated Python 3.11.5+ environment; never repair shared Colab packages')
    pinned = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;\\]+)', Path(lock_path).read_text(), re.M)
    if not {'torch', 'transformers', 'peft', 'bitsandbytes'} <= {name for name, _ in pinned}:
        raise RuntimeError('Expected the complete reviewed model-runtime lock')
    mismatches = []
    for name, expected in pinned:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = 'missing'
        if installed != expected:
            mismatches.append(f'{name}: expected {expected}, found {installed}')
    if mismatches:
        raise RuntimeError('Install the complete hash lock in a fresh venv:\n'+'\n'.join(mismatches))
    return {'python': sys.version.split()[0], 'lock': Path(lock_path).name, 'status': 'versions matched'}

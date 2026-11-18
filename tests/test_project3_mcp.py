import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from project3_mcp_check import check
from project3_mcp import build_server


def test_real_mcp_client_discovery_calls_and_write_exclusion(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'authored-not-a-real-credential')
    (tmp_path/'constraints.md').write_text('Human approval is mandatory. No customer PII.')
    result=asyncio.run(check(tmp_path))
    assert result['budget_total']==3000
    assert result['search']['matches'][0]['source']=='constraints.md'
    assert not (tmp_path/'outside.md').exists()


def test_knowledge_snapshot_rejects_symlink_escape(tmp_path):
    outside=tmp_path/'outside.md'; outside.write_text('not part of this snapshot')
    root=tmp_path/'knowledge'; root.mkdir()
    (root/'escape.md').symlink_to(outside)
    with pytest.raises(ValueError, match='path'):
        build_server(root)

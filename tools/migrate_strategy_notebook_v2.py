"""Reviewed, cell-scoped LangGraph migration; old experiment artifacts are read-only."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'03_llm_agent_system.ipynb'


def main():
    notebook = json.loads(PATH.read_text())
    cells = {c.get('id'): ''.join(c['source']) for c in notebook['cells']}
    original = {c.get('id'): ''.join(c['source']) for c in json.loads(
        subprocess.check_output(['git', 'show', f'HEAD:{PATH.name}'], cwd=ROOT))['cells']}
    baseline_tail = original['a24313e8'].split('\nstrategy_tasks = strategy_cases(KNOWLEDGE_BASE)\n', 1)[1]
    updates = {
        '2cb02c52': original['2cb02c52'].replace('A CrewAI marketing workflow', 'An explicit LangGraph marketing workflow')
            + '\nThe v2 workflow is a new implementation. Historical CrewAI comparisons remain unchanged.\n',
        '4b855404': '''# Isolated patched inference environment; GPU execution is deferred by the user.
import os, sys
from pathlib import Path
from importlib.metadata import version
REPO_ROOT = Path(os.environ.get("PROJECT_REPO", Path.cwd())).resolve()
sys.path.insert(0, str(REPO_ROOT / "tools"))
from training_environment import validate
print(validate(REPO_ROOT / "tools/requirements-inference-v2.hashes.txt"))
''',
        '87bbb7e4': original['87bbb7e4'].replace("os.environ.setdefault('CREWAI_DISABLE_TELEMETRY', 'true')\n", '')
            .replace("'outputs/runs/upgrade-v1'", "'outputs/runs/upgrade-v2'")
            .replace("    agent_model = generator_model", "    if generator_model.config._commit_hash != MODEL_REVISION:\n        raise RuntimeError('Shared model revision differs from the pinned baseline')\n    agent_model = generator_model")
            .replace('            low_cpu_mem_usage=True,', '            trust_remote_code=False,\n            use_safetensors=True,'),
        '13fffca2': '''## 4. Explicit LangGraph Role Workflow (v2)

Research -> plan -> review -> edit is a fixed acyclic graph with typed state.
Each node receives the source brief and the required prior outputs. A shared
budget enforces measured input/output tokens and call count; failed calls stop
the graph, with no hidden retries, delegation, memory, or side effects. Node
traces record prompt/answer hashes and timing. This is orchestration integration,
not evidence that four roles outperform one model. Historical CrewAI scores do
not describe this implementation; new model-backed comparisons are deferred.
''',
        'a24313e8': (ROOT/'tools/project3_strategy_evaluation.py').read_text()
            + '\nstrategy_tasks = strategy_cases(KNOWLEDGE_BASE)\n'
            + baseline_tail.replace('crewai', 'langgraph_roles').replace('crew_', 'role_'),
        'b3a5ffa9': original['b3a5ffa9'].replace('crewai', 'langgraph_roles').replace('crew_', 'role_')
            .replace('PROJECT3_CREW_MEMO', 'PROJECT3_LANGGRAPH_MEMO')
            .replace("'evaluation_version': 'upgrade-v1'", "'evaluation_version': 'upgrade-v2',\n    'workflow_version': 'strategy-langgraph-v2',\n    'langgraph_version': version('langgraph'),\n    'strategy_requirements_sha256': fingerprint(SHARED_REQUIREMENTS)"),
    }
    description = notebook['cells'][12]
    updates[description['id']] = '''## 5. Controlled Evaluation

Routing correctness, harness fallback, execution, safety, and latency are separate.
One-shot, self-review, and LangGraph role arms share maximum aggregate budgets and
source briefs. Keyword coverage remains only a smoke proxy. Blinded content ratings
are a separate step; new v2 model-backed ratings have not been run.
'''
    for identifier, source in updates.items():
        if cells[identifier] not in (source, original[identifier]):
            raise RuntimeError(f'Preserving local cell {identifier}; merge it explicitly')
    changed = []
    for index, cell in enumerate(notebook['cells'], 1):
        if cell.get('id') in updates and ''.join(cell['source']) != updates[cell['id']]:
            cell['source'] = updates[cell['id']].splitlines(keepends=True)
            if cell['cell_type'] == 'code':
                cell.update(outputs=[], execution_count=None)
            changed.append({'cell': index, 'id': cell['id']})
    additions = [
        {'cell_type':'markdown', 'id':'p3-mcp-v2-description', 'metadata':{}, 'source':[
            '## 6. Real MCP Read-Only Integration\n',
            'This is a real SDK stdio client/server check, separate from the in-process manifest and model-routing scores. Search and calculation are available; no write authority or A2A conformance is claimed.\n']},
        {'cell_type':'code', 'id':'p3-mcp-v2-check', 'metadata':{}, 'outputs':[], 'execution_count':None,
         'source': ['from project3_mcp_check import check as check_mcp\n',
                    'mcp_integration = await check_mcp(KB_DIR)\n',
                    '(OUT_DIR / "mcp_integration_v2.json").write_text(json.dumps(mcp_integration, indent=2))\n',
                    'print(mcp_integration)\n']}]
    existing = {cell.get('id') for cell in notebook['cells']}
    for cell in additions:
        if cell['id'] not in existing:
            notebook['cells'].append(cell)
            changed.append({'cell':len(notebook['cells']), 'id':cell['id'], 'operation':'append'})
    PATH.write_text(json.dumps(notebook, indent=1)+'\n')
    print(json.dumps(changed))


if __name__ == '__main__':
    main()

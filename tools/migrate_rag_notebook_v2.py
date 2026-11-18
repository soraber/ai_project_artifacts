"""Cell-scoped conditional HyDE migration; do not replace local notebook edits."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'02_rag_document_qa_system.ipynb'


def main():
    notebook = json.loads(PATH.read_text())
    original = {c.get('id'): ''.join(c['source']) for c in json.loads(
        subprocess.check_output(['git','show',f'HEAD:{PATH.name}'],cwd=ROOT))['cells']}
    evaluation = original['45f0634e'].replace('def ids_at(positions):', '''from conditional_hyde import retrieve as conditional_retrieve, POLICY as HYDE_POLICY
conditional_hyde_trace = []
def conditional_positions(question):
    return conditional_retrieve(question, dense_rank, sparse_rank, reciprocal_rank_fusion,
        rerank, lambda q: generate_batch([hyde_prompt(q)], max_new_tokens=96)[0],
        CANDIDATE_K, ADVANCED_K, conditional_hyde_trace)

def ids_at(positions):''').replace(
        '    "hyde_hybrid_rerank": lambda q: ids_at(hyde_positions(q)),',
        '    "hyde_hybrid_rerank": lambda q: ids_at(hyde_positions(q)),\n'
        '    "conditional_hyde_hybrid_rerank": lambda q: ids_at(conditional_positions(q)),')
    evaluation = evaluation.replace('records, arm_summaries = run_rag_ablation(',
        'conditional_hyde_trace.clear()\nrecords, arm_summaries = run_rag_ablation(')
    evaluation = evaluation.replace('"evaluation_version": "upgrade-v1",',
        '"evaluation_version": "upgrade-v2",\n'
        '    "conditional_hyde_policy": HYDE_POLICY,\n'
        '    "conditional_hyde_trace": conditional_hyde_trace,')
    updates = {
        '67c7857a': '''# Select the isolated, hash-locked inference environment. GPU execution is deferred.
import os, sys
from pathlib import Path
from importlib.metadata import version
REPO_ROOT = Path(os.environ.get("PROJECT_REPO", Path.cwd())).resolve()
sys.path.insert(0, str(REPO_ROOT / "tools"))
from training_environment import validate
print(validate(REPO_ROOT / "tools/requirements-inference-v2.hashes.txt"))
''',
        '3d8e1486': original['3d8e1486'].replace("'outputs/runs/upgrade-v1'", "'outputs/runs/upgrade-v2'"),
        'bdf0279c': original['bdf0279c'].replace('    low_cpu_mem_usage=True,',
            '    trust_remote_code=False,\n    use_safetensors=True,'),
        '67e3a2f8': original['67e3a2f8'] + '''
The v2 comparison adds conditional HyDE as a fifth arm. It skips hypothesis
generation when at least two of the top three dense and positive-scoring lexical
candidates agree. This preregistered rule is a heuristic, not calibrated relevance
or factual confidence. All arms retain the same answer prompt and final context
budget; the probe and optional hypothesis count toward measured latency and tokens.
Warm-up decisions are excluded. New model-backed results are pending.
''',
        '45f0634e': evaluation,
    }
    changed=[]
    for index,cell in enumerate(notebook['cells'],1):
        identifier=cell.get('id')
        if identifier not in updates: continue
        source=''.join(cell['source'])
        if source not in (original[identifier],updates[identifier]):
            raise RuntimeError(f'Preserving local cell {identifier}; merge it explicitly')
        if source!=updates[identifier]:
            cell['source']=updates[identifier].splitlines(keepends=True)
            if cell['cell_type']=='code': cell.update(outputs=[],execution_count=None)
            changed.append({'cell':index,'id':identifier})
    PATH.write_text(json.dumps(notebook,indent=1)+'\n')
    print(json.dumps(changed))


if __name__=='__main__': main()

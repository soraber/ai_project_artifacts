import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from conditional_hyde import gate,retrieve


def test_agreement_and_zero_bm25_scores_do_not_mean_probability():
    assert not gate([(1,.8),(2,.7),(3,.6)],[(1,3),(2,2),(4,1)])['use_hyde']
    assert gate([(1,.8),(2,.7)],[(1,0),(2,0)])['use_hyde']
    with pytest.raises(ValueError):gate([(-1,.5)],[(1,1)])
    with pytest.raises(ValueError):gate([(1,1),(2,1),(3,1),(-1,1)],[(1,1)])


def test_gated_path_saves_exactly_one_hypothesis_call_and_keeps_final_budget():
    generated=[];trace=[]
    def hypothesis(q):generated.append(q);return 'hypothetical data'
    def rerank(q,candidates,k):assert k==2;return candidates[:k]
    dense=lambda q,k:[(1,.9),(2,.8),(3,.7)]
    fuse=lambda rankings,limit:[1,2,3]
    retrieve('q',dense,lambda q,k:[(1,3),(2,2),(3,1)],fuse,rerank,hypothesis,3,2,trace)
    assert generated==[] and not trace[-1]['use_hyde']
    retrieve('q',dense,lambda q,k:[(4,3),(5,2),(6,1)],fuse,rerank,hypothesis,3,2,trace)
    assert generated==['q'] and trace[-1]['use_hyde']


def test_hypothesis_failure_is_retained_not_silently_retried():
    trace=[]
    with pytest.raises(ValueError,match='Empty'):
        retrieve('q',lambda q,k:[(1,.5)],lambda q,k:[(2,1)],lambda rows,k:[1],
                 lambda q,ids,k:ids,lambda q:'',2,1,trace)
    assert trace[-1]['status']=='failed'


def test_reranker_cannot_return_unknown_or_duplicate_ids():
    for result in [[99],[1,1]]:
        with pytest.raises(ValueError,match='reranked'):
            retrieve('q',lambda q,k:[(1,.5)],lambda q,k:[(2,1)],lambda rows,k:[1,2],
                     lambda q,ids,k:result,lambda q:'hypothesis',2,2,[])

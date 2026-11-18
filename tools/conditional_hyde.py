"""Preregistered rank-agreement gate, not a calibrated relevance probability."""
import hashlib
import math

POLICY = {'version':'rank-agreement-v1','top_k':3,'required_shared_ids':2,
          'scope':'heuristic candidate; no held-out threshold tuning or factual-confidence claim'}


def gate(dense, lexical, top_k=3, required_shared_ids=2):
    if not 1 <= required_shared_ids <= top_k:
        raise ValueError('Invalid rank agreement policy')
    def checked(rows, positive_only=False):
        ids=[]
        for index,score in rows:
            if type(index) is not int or index<0 or not math.isfinite(score):
                raise ValueError('Invalid retrieval rank')
            if positive_only and score<=0: continue
            if index not in ids and len(ids)<top_k: ids.append(index)
        return ids
    dense_ids=checked(dense); lexical_ids=checked(lexical,True)
    overlap=len(set(dense_ids)&set(lexical_ids))
    return {'use_hyde':overlap<required_shared_ids,'shared_ids':overlap,
            'dense_count':len(dense_ids),'lexical_count':len(lexical_ids)}


def retrieve(question, dense_rank, sparse_rank, fuse, rerank, hypothetical,
             candidate_k, final_k, trace):
    if not 1 <= final_k <= candidate_k:
        raise ValueError('Invalid retrieval budget')
    dense=dense_rank(question,candidate_k)
    lexical=sparse_rank(question,candidate_k)
    decision=gate(dense,lexical,POLICY['top_k'],POLICY['required_shared_ids'])
    event={'question_sha256':hashlib.sha256(question.encode()).hexdigest(),**decision,'status':'started'}
    trace.append(event)
    try:
        if decision['use_hyde']:
            passage=hypothetical(question)
            if not isinstance(passage,str) or not passage.strip(): raise ValueError('Empty hypothetical passage')
            dense=dense_rank(question+' '+passage,candidate_k)
            gate(dense,lexical)
        candidates=fuse([dense,lexical],candidate_k)
        known={index for index,_ in dense+lexical}
        if len(candidates)>candidate_k or len(set(candidates))!=len(candidates) or any(
            type(index) is not int or index not in known for index in candidates
        ):
            raise ValueError('Invalid fused candidates')
        result=rerank(question,candidates,final_k)
        if len(result)>final_k or len(set(result))!=len(result) or any(
            type(index) is not int or index not in candidates for index in result
        ):
            raise ValueError('Invalid reranked candidates')
        event['status']='complete'
        return result
    except Exception as exc:
        event.update(status='failed',error_type=type(exc).__name__)
        raise

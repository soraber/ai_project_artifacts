import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK = Path(__file__).with_name("02_rag_document_qa_system.ipynb")


def lines(text):
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


nb = json.loads(NOTEBOOK.read_text())
old_cells = nb["cells"]
title = nb["cells"][0]

nb["cells"] = [
    title,
    markdown(
        """
        ## Objective

        Build and measure an end-to-end document-QA RAG system on the public
        **RAG Mini-Wikipedia** benchmark. The experiment compares a dense-retrieval
        baseline with an advanced pipeline using HyDE, dense+sparse reciprocal-rank
        fusion, cross-encoder reranking, evidence-first prompting, and citations.

        Both systems use the same local Llama answer model, isolating the value of
        the retrieval and prompting changes. The final cells compute real metrics,
        save per-example evidence, and provide optional GPT-4o and DeepInfra adapters.
        """
    ),
    code(
        """
        # Lightweight installs preserve the compatible stack already used by Project 1.
        %pip install -q --upgrade-strategy only-if-needed \\
            "datasets>=3,<5" "faiss-cpu>=1.9,<2" "rank-bm25==0.2.2" \\
            "langchain-core>=0.3,<2" "langchain-text-splitters>=0.3,<2" \\
            "bitsandbytes>=0.45" "accelerate>=1"

        from importlib.metadata import version
        for package in ["datasets", "faiss-cpu", "rank-bm25", "transformers", "torch"]:
            print(f"{package:24s} {version(package)}")
        """
    ),
    code(
        """
        import gc
        import json
        import math
        import os
        import pathlib
        import random
        import re
        import statistics
        import time
        from collections import Counter

        import numpy as np
        import pandas as pd
        import torch

        # Standard HTTP downloads are more reliable through the VS Code Colab proxy.
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

        PROJECT_DIR = pathlib.Path('/content/rag_document_qa')
        CACHE_DIR = PROJECT_DIR / 'cache'
        OUT_DIR = PROJECT_DIR / 'outputs'
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Seed 42 was used for prompt development; 2026 is the untouched final holdout.
        SEED = 2026
        random.seed(SEED)
        np.random.seed(SEED)

        DATASET_NAME = 'rag-datasets/rag-mini-wikipedia'
        DATASET_REVISION = 'ec8e240df1b1eb7171cb3e5dcf9462f6569f4544'
        EMBED_MODEL = 'intfloat/e5-small-v2'
        RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        GENERATOR_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
        EVAL_SIZE = 60
        BASELINE_K = 3
        ADVANCED_K = 5
        CANDIDATE_K = 20

        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA runtime is required. Select the connected A100 Colab kernel.')
        DEVICE = torch.device('cuda:0')
        print('GPU:', torch.cuda.get_device_name(0))

        # Release large Project 1 models if both notebooks share the same Colab runtime.
        for variable_name in ['model', 'base_model', 'trainer']:
            if variable_name in globals():
                del globals()[variable_name]
        gc.collect()
        torch.cuda.empty_cache()
        """
    ),
    markdown(
        """
        ## 1. Load and cache a real benchmark

        The dataset contains roughly 3,200 Wikipedia passages and 918 questions.
        It is public (CC BY 3.0), small enough for a portfolio notebook, and large
        enough to exercise real retrieval. A pinned dataset revision makes later
        runs reproducible. `save_to_disk` means the network download happens once.
        """
    ),
    code(
        """
        from datasets import load_dataset, load_from_disk
        from huggingface_hub import hf_hub_download, list_repo_files

        CORPUS_CACHE = CACHE_DIR / 'mini_wikipedia_corpus'
        QA_CACHE = CACHE_DIR / 'mini_wikipedia_qa'

        if CORPUS_CACHE.exists() and QA_CACHE.exists():
            corpus_ds = load_from_disk(str(CORPUS_CACHE))
            qa_ds = load_from_disk(str(QA_CACHE))
            source = 'local cache'
        else:
            repo_files = list_repo_files(
                DATASET_NAME, repo_type='dataset', revision=DATASET_REVISION
            )
            corpus_files = sorted(
                path for path in repo_files
                if path.startswith('text-corpus/passages/') and path.endswith('.parquet')
            )
            qa_files = sorted(
                path for path in repo_files
                if path.startswith('question-answer/test/') and path.endswith('.parquet')
            )
            if not corpus_files or not qa_files:
                raise RuntimeError('Could not locate the pinned corpus and QA parquet files.')

            def download_parquet_files(paths):
                return [
                    hf_hub_download(
                        repo_id=DATASET_NAME,
                        filename=path,
                        repo_type='dataset',
                        revision=DATASET_REVISION,
                        cache_dir=str(CACHE_DIR / 'huggingface'),
                    )
                    for path in paths
                ]

            corpus_ds = load_dataset(
                'parquet', data_files=download_parquet_files(corpus_files), split='train'
            )
            qa_ds = load_dataset(
                'parquet', data_files=download_parquet_files(qa_files), split='train'
            )
            corpus_ds.save_to_disk(str(CORPUS_CACHE))
            qa_ds.save_to_disk(str(QA_CACHE))
            source = 'Hugging Face download (now cached)'

        print('Loaded from:', source)
        print('Corpus rows:', len(corpus_ds), '| QA rows:', len(qa_ds))
        print('Corpus columns:', corpus_ds.column_names)
        print('QA columns:', qa_ds.column_names)
        """
    ),
    code(
        """
        from langchain_core.documents import Document

        documents = []
        for row_index, row in enumerate(corpus_ds):
            passage = str(row.get('passage', '')).strip()
            if not passage:
                continue
            passage_id = str(row.get('id', row_index))
            documents.append(
                Document(
                    page_content=passage,
                    metadata={'source': 'mini_wikipedia', 'passage_id': passage_id},
                )
            )

        assert documents, 'No corpus documents were created.'
        print('LangChain documents:', len(documents))
        print(documents[0].metadata, documents[0].page_content[:180])
        """
    ),
    markdown(
        """
        ## 2. Structure-aware chunking

        Recursive splitting prefers paragraph and sentence boundaries before hard
        character cuts. A 900-character window retains enough factual context while
        a 120-character overlap protects facts near boundaries. Stable chunk IDs
        support traceable citations and repeatable evaluation.
        """
    ),
    code(
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
            separators=['\\n\\n', '\\n', '. ', ' ', ''],
        )

        chunks = []
        for document in documents:
            for local_index, chunk in enumerate(splitter.split_documents([document])):
                chunk.metadata['chunk_id'] = (
                    f"wiki::{document.metadata['passage_id']}::{local_index}"
                )
                chunks.append(chunk)

        lengths = [len(chunk.page_content) for chunk in chunks]
        print('Chunks:', len(chunks))
        print('Characters per chunk: median', int(np.median(lengths)), 'p95', int(np.percentile(lengths, 95)))
        assert len({chunk.metadata['chunk_id'] for chunk in chunks}) == len(chunks)
        """
    ),
    markdown(
        """
        ## 3. Build dense and sparse indexes

        E5-small-v2 is a strong, efficient retrieval encoder. FAISS provides exact
        inner-product search over normalized vectors, while BM25 captures names,
        numbers, and rare terms that dense retrieval can miss. Embeddings are cached
        so reruns skip the most expensive indexing step.
        """
    ),
    code(
        """
        import faiss
        import torch.nn.functional as F
        from rank_bm25 import BM25Okapi
        from transformers import AutoModel, AutoTokenizer

        def lexical_tokens(text):
            return re.findall(r"[a-z0-9]+", text.lower())

        class E5Encoder:
            def __init__(self, model_name, device):
                self.device = device
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(device).eval()

            @torch.inference_mode()
            def encode(self, texts, prefix, batch_size=64):
                vectors = []
                for start in range(0, len(texts), batch_size):
                    batch = [prefix + text for text in texts[start:start + batch_size]]
                    encoded = self.tokenizer(
                        batch, padding=True, truncation=True, max_length=512, return_tensors='pt'
                    ).to(self.device)
                    hidden = self.model(**encoded).last_hidden_state
                    mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    vectors.append(F.normalize(pooled, p=2, dim=1).float().cpu().numpy())
                return np.concatenate(vectors).astype('float32')

        encoder = E5Encoder(EMBED_MODEL, DEVICE)
        chunk_texts = [chunk.page_content for chunk in chunks]
        embedding_cache = CACHE_DIR / 'e5_small_v2_chunk_embeddings.npy'

        if embedding_cache.exists():
            chunk_embeddings = np.load(embedding_cache)
            if chunk_embeddings.shape[0] != len(chunks):
                raise RuntimeError('Cached embeddings do not match the current chunk count.')
            print('Loaded embeddings from cache:', embedding_cache)
        else:
            chunk_embeddings = encoder.encode(chunk_texts, prefix='passage: ')
            np.save(embedding_cache, chunk_embeddings)
            print('Created and cached embeddings:', embedding_cache)

        dense_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
        dense_index.add(chunk_embeddings)
        tokenized_corpus = [lexical_tokens(text) for text in chunk_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        print('FAISS vectors:', dense_index.ntotal, '| dimensions:', dense_index.d)
        """
    ),
    markdown(
        """
        ## 4. Shared local answer model

        The connected runtime already contains the Meta Llama 3 8B weights used by
        Project 1. Loading those cached weights in NF4 4-bit mode avoids another
        multi-gigabyte transfer and provides a stronger local generator. Using one
        generator for both systems controls a major confounder: score differences
        can be attributed to retrieval and prompt design rather than model size.
        """
    ),
    code(
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        generator_quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        try:
            generator_tokenizer = AutoTokenizer.from_pretrained(
                GENERATOR_MODEL, local_files_only=True
            )
            generator_model = AutoModelForCausalLM.from_pretrained(
                GENERATOR_MODEL,
                local_files_only=True,
                quantization_config=generator_quantization,
                device_map='auto',
                dtype=compute_dtype,
                attn_implementation='sdpa',
                low_cpu_mem_usage=True,
            ).eval()
        except OSError as exc:
            raise RuntimeError(
                'The cached Project 1 Llama weights were not found. Run Project 1 in this '
                'runtime or pre-download the model in the Colab UI, then rerun Cell 13.'
            ) from exc

        if generator_tokenizer.pad_token_id is None:
            generator_tokenizer.pad_token = generator_tokenizer.eos_token
        generator_tokenizer.padding_side = 'left'
        generator_model.config.use_cache = True

        @torch.inference_mode()
        def generate_batch(prompts, max_new_tokens=96, batch_size=8):
            answers = []
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start:start + batch_size]
                rendered = [
                    generator_tokenizer.apply_chat_template(
                        [
                            {'role': 'system', 'content': 'You are a precise document QA assistant.'},
                            {'role': 'user', 'content': prompt},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for prompt in batch_prompts
                ]
                encoded = generator_tokenizer(
                    rendered, padding=True, truncation=True, max_length=3072, return_tensors='pt'
                ).to(DEVICE)
                generated = generator_model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=generator_tokenizer.pad_token_id,
                )
                new_tokens = generated[:, encoded['input_ids'].shape[1]:]
                answers.extend(generator_tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            return [answer.strip() for answer in answers]

        print('Loaded generator:', GENERATOR_MODEL)
        """
    ),
    markdown(
        """
        ## 5. Baseline and advanced retrieval

        The baseline uses a raw-question E5 search. The advanced retriever creates a
        hypothetical answer passage (HyDE), searches both dense and BM25 indexes,
        combines rankings with reciprocal-rank fusion, and reranks the top candidates
        with a query-document cross-encoder. This stack balances semantic recall,
        exact-term recall, and final precision.
        """
    ),
    code(
        """
        from transformers import AutoModelForSequenceClassification

        rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL).to(DEVICE).eval()

        def dense_rank(query, k):
            query_vector = encoder.encode([query], prefix='query: ')
            scores, indices = dense_index.search(query_vector, k)
            return [(int(index), float(score)) for index, score in zip(indices[0], scores[0])]

        def sparse_rank(query, k):
            scores = bm25.get_scores(lexical_tokens(query))
            indices = np.argsort(scores)[::-1][:k]
            return [(int(index), float(scores[index])) for index in indices]

        def reciprocal_rank_fusion(rankings, limit, rrf_k=60):
            fused = Counter()
            for ranking in rankings:
                for rank, (index, _) in enumerate(ranking, start=1):
                    fused[index] += 1.0 / (rrf_k + rank)
            return [index for index, _ in fused.most_common(limit)]

        @torch.inference_mode()
        def rerank(question, candidate_indices, k):
            pairs = [[question, chunk_texts[index]] for index in candidate_indices]
            encoded = rerank_tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors='pt'
            ).to(DEVICE)
            scores = rerank_model(**encoded).logits.squeeze(-1).float().cpu().numpy()
            order = np.argsort(scores)[::-1][:k]
            return [candidate_indices[int(position)] for position in order]

        def baseline_retrieve(question):
            return [index for index, _ in dense_rank(question, BASELINE_K)]

        def advanced_retrieve(question, hypothetical_passage):
            dense_query = f"{question} {hypothetical_passage}"
            candidates = reciprocal_rank_fusion(
                [dense_rank(dense_query, CANDIDATE_K), sparse_rank(question, CANDIDATE_K)],
                limit=CANDIDATE_K,
            )
            return rerank(question, candidates, ADVANCED_K)

        def format_context(indices):
            return '\\n\\n'.join(
                f"[{chunks[index].metadata['chunk_id']}] {chunks[index].page_content}"
                for index in indices
            )

        def hyde_prompt(question):
            return (
                'Write a short Wikipedia-style passage that would answer the question below. '
                'Include likely entities, dates, and terminology. Return only the hypothetical passage.\\n\\n'
                f'Question: {question}'
            )

        def baseline_answer_prompt(question, indices):
            return (
                'Answer the question using only the supplied context. If the answer is absent, say '
                '"Insufficient evidence." Give a concise answer and cite supporting chunk IDs in brackets.\\n\\n'
                f'Question: {question}\\n\\nContext:\\n{format_context(indices)}'
            )

        def advanced_answer_prompt(question, indices):
            return (
                'Select the strongest supporting evidence internally and answer only what the context '
                'supports. Return only the shortest correct answer phrase or sentence followed by the '
                'supporting chunk ID in brackets. Do not include analysis, a preamble, quoted context, '
                'or a restatement of the question. If evidence is insufficient, return exactly '
                '"Insufficient evidence."\\n\\n'
                f'Question: {question}\\n\\nRanked evidence:\\n{format_context(indices)}'
            )

        print('Retriever components ready: dense, BM25, RRF, HyDE, cross-encoder reranker')
        """
    ),
    markdown(
        """
        ## 6. Reproducible evaluation

        The official QA split does not provide gold passage IDs, so retrieval is
        evaluated with **answer-hit@k**: whether a normalized reference answer occurs
        in any retrieved chunk. We also report reciprocal rank, answer token F1,
        citation validity, and latency. The deterministic sample only includes
        questions whose answer text occurs somewhere in the supplied corpus.

        The transparent composite score weights answer F1 (45%), answer-hit (25%),
        reciprocal rank (15%), and citation validity (15%). The claimed improvement
        is computed from saved predictions; no target percentage is hardcoded.
        """
    ),
    code(
        """
        def normalize_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\\s]', ' ', text)
            return ' '.join(text.split())

        def answer_text(value):
            if isinstance(value, (list, tuple)):
                return str(value[0]) if value else ''
            return str(value)

        def strip_citations(text):
            return re.sub(r'\\[wiki::[^\\]]+\\]', ' ', str(text), flags=re.IGNORECASE)

        def token_f1(prediction, reference):
            prediction_tokens = normalize_text(strip_citations(prediction)).split()
            reference_tokens = normalize_text(reference).split()
            if not prediction_tokens or not reference_tokens:
                return float(prediction_tokens == reference_tokens)
            common = Counter(prediction_tokens) & Counter(reference_tokens)
            overlap = sum(common.values())
            if overlap == 0:
                return 0.0
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(reference_tokens)
            return 2 * precision * recall / (precision + recall)

        normalized_chunks = [normalize_text(text) for text in chunk_texts]
        normalized_corpus = '\\n'.join(normalized_chunks)
        eligible = []
        for row in qa_ds:
            question = str(row.get('question', '')).strip()
            reference = answer_text(row.get('answer', '')).strip()
            normalized_answer = normalize_text(reference)
            if (
                question
                and len(normalized_answer) >= 3
                and normalized_answer not in {'yes', 'no', 'true', 'false'}
                and normalized_answer in normalized_corpus
            ):
                eligible.append({'question': question, 'reference': reference})

        if len(eligible) < EVAL_SIZE:
            raise RuntimeError(f'Only {len(eligible)} answer-grounded examples are available.')
        eval_examples = random.Random(SEED).sample(eligible, EVAL_SIZE)
        questions = [example['question'] for example in eval_examples]
        references = [example['reference'] for example in eval_examples]
        print('Eligible QA:', len(eligible), '| deterministic evaluation sample:', len(eval_examples))

        baseline_started = time.perf_counter()
        baseline_indices = [baseline_retrieve(question) for question in questions]
        baseline_retrieval_seconds = time.perf_counter() - baseline_started

        hyde_started = time.perf_counter()
        hypothetical_passages = generate_batch(
            [hyde_prompt(question) for question in questions], max_new_tokens=96
        )
        hyde_seconds = time.perf_counter() - hyde_started

        advanced_started = time.perf_counter()
        advanced_indices = [
            advanced_retrieve(question, hypothetical)
            for question, hypothetical in zip(questions, hypothetical_passages)
        ]
        advanced_retrieval_seconds = time.perf_counter() - advanced_started

        baseline_generation_started = time.perf_counter()
        baseline_answers = generate_batch(
            [baseline_answer_prompt(q, idx) for q, idx in zip(questions, baseline_indices)],
            max_new_tokens=96,
        )
        baseline_generation_seconds = time.perf_counter() - baseline_generation_started

        advanced_generation_started = time.perf_counter()
        advanced_answers = generate_batch(
            [advanced_answer_prompt(q, idx) for q, idx in zip(questions, advanced_indices)],
            max_new_tokens=96,
        )
        advanced_generation_seconds = time.perf_counter() - advanced_generation_started

        def retrieval_metrics(reference, indices):
            target = normalize_text(reference)
            ranks = [rank for rank, index in enumerate(indices, start=1) if target in normalized_chunks[index]]
            return (1.0 if ranks else 0.0, 1.0 / ranks[0] if ranks else 0.0)

        def citation_validity(answer, indices):
            cited = re.findall(r'\\[(wiki::[^\\]]+)\\]', answer, flags=re.IGNORECASE)
            if not cited:
                return 0.0
            valid = {chunks[index].metadata['chunk_id'].lower() for index in indices}
            return sum(citation.lower() in valid for citation in cited) / len(cited)

        records = []
        for position, example in enumerate(eval_examples):
            for system, indices, answer in [
                ('baseline', baseline_indices[position], baseline_answers[position]),
                ('advanced', advanced_indices[position], advanced_answers[position]),
            ]:
                hit, reciprocal_rank = retrieval_metrics(example['reference'], indices)
                records.append({
                    'example_id': position,
                    'system': system,
                    'question': example['question'],
                    'reference': example['reference'],
                    'answer': answer,
                    'retrieved_chunk_ids': json.dumps(
                        [chunks[index].metadata['chunk_id'] for index in indices]
                    ),
                    'answer_hit': hit,
                    'reciprocal_rank': reciprocal_rank,
                    'token_f1': token_f1(answer, example['reference']),
                    'citation_validity': citation_validity(answer, indices),
                })

        results = pd.DataFrame(records)
        summary = results.groupby('system')[
            ['answer_hit', 'reciprocal_rank', 'token_f1', 'citation_validity']
        ].mean()
        summary['composite_score'] = (
            0.25 * summary['answer_hit']
            + 0.15 * summary['reciprocal_rank']
            + 0.45 * summary['token_f1']
            + 0.15 * summary['citation_validity']
        )
        summary['retrieval_seconds_per_question'] = np.nan
        summary.loc['baseline', 'retrieval_seconds_per_question'] = (
            baseline_retrieval_seconds / EVAL_SIZE
        )
        summary.loc['advanced', 'retrieval_seconds_per_question'] = (
            (hyde_seconds + advanced_retrieval_seconds) / EVAL_SIZE
        )
        summary['generation_seconds_per_question'] = np.nan
        summary.loc['baseline', 'generation_seconds_per_question'] = (
            baseline_generation_seconds / EVAL_SIZE
        )
        summary.loc['advanced', 'generation_seconds_per_question'] = (
            advanced_generation_seconds / EVAL_SIZE
        )

        baseline_score = float(summary.loc['baseline', 'composite_score'])
        advanced_score = float(summary.loc['advanced', 'composite_score'])
        relative_lift = 100.0 * (advanced_score - baseline_score) / max(baseline_score, 1e-9)

        results.to_csv(OUT_DIR / 'project2_eval_results.csv', index=False)
        summary.reset_index().to_csv(OUT_DIR / 'project2_metrics.csv', index=False)
        run_summary = {
            'dataset': DATASET_NAME,
            'dataset_revision': DATASET_REVISION,
            'corpus_rows': len(corpus_ds),
            'qa_rows': len(qa_ds),
            'chunks': len(chunks),
            'evaluation_examples': EVAL_SIZE,
            'evaluation_seed': SEED,
            'generator_model': GENERATOR_MODEL,
            'embedding_model': EMBED_MODEL,
            'reranker_model': RERANK_MODEL,
            'baseline_composite': baseline_score,
            'advanced_composite': advanced_score,
            'relative_improvement_percent': relative_lift,
            'metrics': summary.reset_index().to_dict(orient='records'),
        }
        (OUT_DIR / 'project2_summary.json').write_text(json.dumps(run_summary, indent=2))

        display(summary.round(4))
        print(f'Advanced vs baseline composite improvement: {relative_lift:.2f}%')
        sample_export = []
        for example_id in range(min(3, EVAL_SIZE)):
            pair = results[results['example_id'] == example_id].set_index('system')
            sample_export.append({
                'question': pair.loc['baseline', 'question'],
                'reference': pair.loc['baseline', 'reference'],
                'baseline_answer': pair.loc['baseline', 'answer'],
                'advanced_answer': pair.loc['advanced', 'answer'],
            })
        print('PROJECT2_SUMMARY_JSON=' + json.dumps(run_summary))
        print('PROJECT2_SAMPLES_JSON=' + json.dumps(sample_export))
        print('Saved evaluation artifacts to:', OUT_DIR)
        """
    ),
    markdown(
        """
        ## 7. Optional provider benchmark: GPT-4o, LLaMA, and Mistral

        This cell reuses the exact advanced retrieved contexts for a fair generator
        comparison. It is disabled by default to prevent accidental API charges.
        Keys are read from environment variables or hidden `getpass` prompts and are
        never written to the notebook. DeepInfra exposes an OpenAI-compatible API.
        Set the model IDs to currently available DeepInfra deployments if needed.
        """
    ),
    code(
        """
        RUN_PROVIDER_BENCHMARKS = False
        PROVIDER_EVAL_SIZE = 10

        provider_manifest = pd.DataFrame([
            {
                'provider': 'Local',
                'model': GENERATOR_MODEL,
                'status': 'completed in main evaluation',
            },
            {
                'provider': 'OpenAI',
                'model': os.getenv('OPENAI_MODEL', 'gpt-4o'),
                'status': 'set RUN_PROVIDER_BENCHMARKS=True and supply OPENAI_API_KEY',
            },
            {
                'provider': 'DeepInfra',
                'model': os.getenv('DEEPINFRA_LLAMA_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
                'status': 'set RUN_PROVIDER_BENCHMARKS=True and supply DEEPINFRA_API_KEY',
            },
            {
                'provider': 'DeepInfra',
                'model': os.getenv('DEEPINFRA_MISTRAL_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3'),
                'status': 'set RUN_PROVIDER_BENCHMARKS=True and supply DEEPINFRA_API_KEY',
            },
        ])

        if RUN_PROVIDER_BENCHMARKS:
            %pip install -q --upgrade-strategy only-if-needed "openai>=1,<3"
            from getpass import getpass
            from openai import OpenAI

            def obtain_secret(name):
                value = os.getenv(name)
                if not value:
                    value = getpass(f'{name} (hidden): ').strip()
                return value

            provider_specs = [
                ('OpenAI', os.getenv('OPENAI_MODEL', 'gpt-4o'), None, obtain_secret('OPENAI_API_KEY')),
                (
                    'DeepInfra-LLaMA',
                    os.getenv('DEEPINFRA_LLAMA_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
                    'https://api.deepinfra.com/v1/openai',
                    obtain_secret('DEEPINFRA_API_KEY'),
                ),
                (
                    'DeepInfra-Mistral',
                    os.getenv('DEEPINFRA_MISTRAL_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3'),
                    'https://api.deepinfra.com/v1/openai',
                    os.getenv('DEEPINFRA_API_KEY'),
                ),
            ]
            provider_records = []
            benchmark_prompts = [
                advanced_answer_prompt(q, idx)
                for q, idx in zip(questions[:PROVIDER_EVAL_SIZE], advanced_indices[:PROVIDER_EVAL_SIZE])
            ]
            for provider, model_name, base_url, api_key in provider_specs:
                client = OpenAI(api_key=api_key, base_url=base_url)
                for i, prompt in enumerate(benchmark_prompts):
                    started = time.perf_counter()
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0,
                        max_tokens=96,
                    )
                    answer = response.choices[0].message.content.strip()
                    provider_records.append({
                        'provider': provider,
                        'model': model_name,
                        'example_id': i,
                        'answer': answer,
                        'token_f1': token_f1(answer, references[i]),
                        'citation_validity': citation_validity(answer, advanced_indices[i]),
                        'latency_seconds': time.perf_counter() - started,
                    })
            provider_results = pd.DataFrame(provider_records)
            provider_results.to_csv(OUT_DIR / 'provider_benchmark.csv', index=False)
            display(provider_results.groupby(['provider', 'model'])[
                ['token_f1', 'citation_validity', 'latency_seconds']
            ].mean().round(4))
        else:
            display(provider_manifest)
            print('Provider calls skipped; the local baseline-vs-advanced evaluation is complete.')
        """
    ),
]

# Preserve execution state for cells whose source did not change. This keeps user
# results intact when a targeted cell is corrected and the updater is rerun.
for index, new_cell in enumerate(nb["cells"]):
    if index >= len(old_cells):
        continue
    old_cell = old_cells[index]
    if (
        old_cell.get("cell_type") == new_cell.get("cell_type")
        and old_cell.get("source") == new_cell.get("source")
    ):
        nb["cells"][index] = old_cell

NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Updated {NOTEBOOK} (cells 2-19; cell 1 preserved).")

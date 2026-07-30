import json
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "03_llm_agent_system.ipynb"


def lines(text):
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


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
title = old_cells[0]

nb["cells"] = [
    title,
    markdown(
        """
        ## Objective

        Build a portfolio-grade local agent system with two complementary layers:

        1. An auditable LangChain search agent that selects typed tools, validates
           arguments, cites local evidence, and blocks unapproved writes.
        2. A CrewAI marketing workflow with researcher, strategist, critic, and
           editor roles operating on the same grounded business brief.

        The notebook implements MCP-style input/output schemas, permission
        annotations, execution traces, deterministic safety gates, and a measured
        evaluation rather than a checklist with placeholder values.
        """
    ),
    code(
        """
        %pip install -q --upgrade-strategy only-if-needed \\
            "langchain>=1,<2" "crewai>=1" "rank-bm25==0.2.2" \\
            "pydantic>=2,<3" "pandas>=2" "bitsandbytes>=0.45" "accelerate>=1"

        from importlib.metadata import version
        for package in ['langchain', 'crewai', 'rank-bm25', 'pydantic', 'transformers', 'torch']:
            print(f'{package:18s} {version(package)}')
        """
    ),
    code(
        """
        import gc
        import json
        import os
        import pathlib
        import random
        import re
        import statistics
        import time
        from collections import Counter
        from typing import Any, Literal, Optional

        import numpy as np
        import pandas as pd
        import torch
        from pydantic import BaseModel, Field, ValidationError

        os.environ.setdefault('CREWAI_DISABLE_TELEMETRY', 'true')
        os.environ.setdefault('OTEL_SDK_DISABLED', 'true')
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

        PROJECT_DIR = pathlib.Path('/content/llm_agent_system')
        KB_DIR = PROJECT_DIR / 'knowledge_base'
        OUT_DIR = PROJECT_DIR / 'outputs'
        KB_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        SEED = 2026
        random.seed(SEED)
        np.random.seed(SEED)

        KNOWLEDGE_BASE = {
            'product.md': '''
        # RelayDesk AI
        RelayDesk AI is a workflow assistant for small ecommerce support teams.
        It drafts ticket replies, retrieves inventory and policy answers, and creates
        campaign briefs. It does not send customer messages or publish campaigns
        without human approval. Supported integrations are Shopify, Zendesk, and Slack.
        ''',
            'audience.md': '''
        # Audience and pain points
        Primary buyers are ecommerce founders and support operations leads at teams
        with 5 to 50 employees. Their main problems are repetitive support questions,
        slow first responses, inconsistent policy answers, and limited time for
        campaign planning. Users value fast setup, auditability, and approval controls.
        ''',
            'constraints.md': '''
        # Launch constraints
        The first launch is a two-region, 30-day pilot. Marketing spend must remain
        below $12,000. Every customer-facing AI output requires human approval.
        The team must avoid unsupported ROI claims and must not upload customer PII
        to public tools. Paid media is allowed only after the first two pilot weeks.
        ''',
            'metrics.md': '''
        # Success metrics
        Track ticket deflection rate, median first-response time, answer correction
        rate, qualified demos, trial activation, customer acquisition cost, and the
        percentage of AI drafts approved without edits. Targets must be labeled as
        proposed goals until validated by pilot data.
        ''',
        }
        for filename, content in KNOWLEDGE_BASE.items():
            (KB_DIR / filename).write_text(content.strip() + '\\n')

        if not torch.cuda.is_available():
            raise RuntimeError('Select the connected Colab A100 kernel before continuing.')
        DEVICE = torch.device('cuda:0')
        print('GPU:', torch.cuda.get_device_name(0))

        # Reuse Project 2's model when both notebooks share the same live kernel.
        if 'generator_model' in globals() and 'generator_tokenizer' in globals():
            agent_model = generator_model
            agent_tokenizer = generator_tokenizer
            MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
            model_source = 'reused Project 2 model in memory'
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            try:
                agent_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
                agent_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    local_files_only=True,
                    quantization_config=quantization,
                    device_map='auto',
                    dtype=compute_dtype,
                    attn_implementation='sdpa',
                    low_cpu_mem_usage=True,
                ).eval()
            except OSError as exc:
                raise RuntimeError(
                    'Cached Llama weights were not found. Run Project 2 in this kernel first, '
                    'then rerun Cell 4.'
                ) from exc
            model_source = 'loaded cached Project 2 weights'

        if agent_tokenizer.pad_token_id is None:
            agent_tokenizer.pad_token = agent_tokenizer.eos_token
        agent_tokenizer.padding_side = 'left'
        agent_model.config.use_cache = True
        MODEL_DEVICE = next(agent_model.parameters()).device

        local_generation_calls = 0

        @torch.inference_mode()
        def generate_local(prompt, max_new_tokens=192, temperature=0.0):
            global local_generation_calls
            local_generation_calls += 1
            rendered = agent_tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': 'You are a precise, grounded AI agent.'},
                    {'role': 'user', 'content': str(prompt)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = agent_tokenizer(
                rendered, truncation=True, max_length=7168, return_tensors='pt'
            ).to(MODEL_DEVICE)
            generated = agent_model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=agent_tokenizer.pad_token_id,
            )
            new_tokens = generated[:, encoded['input_ids'].shape[1]:]
            return agent_tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

        print('Local model:', MODEL_NAME, '|', model_source)
        print('Knowledge-base files:', sorted(KNOWLEDGE_BASE))
        """
    ),
    markdown(
        """
        ## 1. Typed local tools and safety boundary

        Tool code, not the LLM, owns data access and side effects. Pydantic validates
        every argument. Search and budget tools are read-only. The report writer
        requires explicit approval, restricts filenames to the output directory,
        and rejects path traversal. Every call emits a structured trace event.
        """
    ),
    code(
        """
        from langchain_core.tools import StructuredTool
        from rank_bm25 import BM25Okapi

        def lexical_tokens(text):
            return re.findall(r'[a-z0-9]+', str(text).lower())

        kb_documents = [
            {'source': path.name, 'text': path.read_text().strip()}
            for path in sorted(KB_DIR.glob('*.md'))
        ]
        bm25 = BM25Okapi([lexical_tokens(document['text']) for document in kb_documents])
        trace_events = []

        class SearchInput(BaseModel):
            query: str = Field(min_length=2, max_length=300)
            top_k: int = Field(default=2, ge=1, le=3)

        class SearchHit(BaseModel):
            source: str
            score: float
            snippet: str

        class SearchOutput(BaseModel):
            query: str
            results: list[SearchHit]

        class BudgetInput(BaseModel):
            channel: str = Field(min_length=2, max_length=80)
            weeks: int = Field(ge=1, le=12)
            weekly_budget: float = Field(gt=0, le=12000)

        class BudgetOutput(BaseModel):
            channel: str
            weeks: int
            weekly_budget: float
            total_budget: float
            within_launch_cap: bool

        class SaveReportInput(BaseModel):
            filename: str = Field(min_length=3, max_length=80)
            content: str = Field(min_length=1, max_length=12000)
            confirm_write: bool = False

        class SaveReportOutput(BaseModel):
            status: Literal['saved', 'blocked']
            reason: Optional[str] = None
            saved_to: Optional[str] = None
            bytes: int = 0

        def traced(tool_name, arguments, operation):
            started = time.perf_counter()
            try:
                result = operation()
                status = result.get('status', 'ok') if isinstance(result, dict) else 'ok'
                return result
            except Exception as exc:
                status = 'error'
                raise
            finally:
                trace_events.append({
                    'tool': tool_name,
                    'arguments': arguments,
                    'status': status,
                    'latency_ms': 1000 * (time.perf_counter() - started),
                })

        def search_knowledge_base(query: str, top_k: int = 2):
            arguments = {'query': query, 'top_k': top_k}
            def operation():
                scores = bm25.get_scores(lexical_tokens(query))
                ranked = np.argsort(scores)[::-1][:top_k]
                hits = []
                for index in ranked:
                    if scores[index] <= 0:
                        continue
                    document = kb_documents[int(index)]
                    hits.append({
                        'source': document['source'],
                        'score': round(float(scores[index]), 4),
                        'snippet': document['text'][:700],
                    })
                return {'query': query, 'results': hits}
            return traced('search_knowledge_base', arguments, operation)

        def calculate_campaign_budget(channel: str, weeks: int, weekly_budget: float):
            arguments = {'channel': channel, 'weeks': weeks, 'weekly_budget': weekly_budget}
            def operation():
                total = round(weeks * weekly_budget, 2)
                return {
                    'channel': channel,
                    'weeks': weeks,
                    'weekly_budget': weekly_budget,
                    'total_budget': total,
                    'within_launch_cap': total <= 12000,
                }
            return traced('calculate_campaign_budget', arguments, operation)

        def save_workspace_report(filename: str, content: str, confirm_write: bool = False):
            arguments = {'filename': filename, 'content_chars': len(content), 'confirm_write': confirm_write}
            def operation():
                if not confirm_write:
                    return {'status': 'blocked', 'reason': 'explicit write approval is required', 'bytes': 0}
                candidate = pathlib.Path(filename)
                if candidate.name != filename or candidate.suffix.lower() != '.md':
                    return {'status': 'blocked', 'reason': 'only simple .md filenames are allowed', 'bytes': 0}
                target = OUT_DIR / candidate.name
                target.write_text(content)
                return {'status': 'saved', 'saved_to': str(target), 'bytes': target.stat().st_size}
            return traced('save_workspace_report', arguments, operation)

        search_tool = StructuredTool.from_function(
            func=search_knowledge_base,
            name='search_knowledge_base',
            description='Search the local RelayDesk business knowledge base and return source-tagged evidence.',
            args_schema=SearchInput,
        )
        budget_tool = StructuredTool.from_function(
            func=calculate_campaign_budget,
            name='calculate_campaign_budget',
            description='Calculate a channel budget and check it against the $12,000 launch cap.',
            args_schema=BudgetInput,
        )
        save_tool = StructuredTool.from_function(
            func=save_workspace_report,
            name='save_workspace_report',
            description='Save an approved Markdown report inside the project output directory.',
            args_schema=SaveReportInput,
        )
        LANGCHAIN_TOOLS = {tool.name: tool for tool in [search_tool, budget_tool, save_tool]}
        print('LangChain tools:', list(LANGCHAIN_TOOLS))
        print(search_tool.invoke({'query': 'primary buyers and approval controls', 'top_k': 2}))
        """
    ),
    markdown(
        """
        ## 2. Auditable LangChain agent controller

        A local Llama planner selects one typed action as JSON. The controller
        validates the decision, retries malformed JSON once, and uses a conservative
        deterministic fallback. Search answers are generated only from returned
        evidence and must cite source filenames. Side-effect permission is injected
        by the controller rather than trusted to model output.
        """
    ),
    code(
        """
        from langchain_core.runnables import RunnableLambda

        class AgentDecision(BaseModel):
            tool: Literal['search_knowledge_base', 'calculate_campaign_budget', 'save_workspace_report']
            arguments: dict[str, Any]

        def extract_json_object(text):
            match = re.search(r'\\{.*\\}', str(text), flags=re.S)
            if not match:
                raise ValueError('No JSON object found')
            return json.loads(match.group(0))

        ROUTER_PROMPT = '''
        Select exactly one tool for the user task and return JSON only:
        {{"tool":"TOOL_NAME","arguments":{{...}}}}

        Tools:
        - search_knowledge_base(query, top_k): questions about the product, audience,
          policies, constraints, metrics, or unsupported business facts.
        - calculate_campaign_budget(channel, weeks, weekly_budget): arithmetic budget requests.
        - save_workspace_report(filename, content, confirm_write): explicit requests to save a report.

        Never claim write approval. Set confirm_write to false; the controller owns permission.
        For search, preserve the important query terms. For save, use the requested filename and content.

        User task: {task}
        '''

        def fallback_decision(task):
            lower = task.lower()
            if any(token in lower for token in ['calculate', 'budget', 'per week', 'weekly']):
                weeks = int((re.search(r'(\\d+)\\s*week', lower) or [None, 1])[1])
                money_match = re.search(r'\\$?([\\d,]+(?:\\.\\d+)?)\\s*(?:per|/)\\s*week', lower)
                weekly = float(money_match.group(1).replace(',', '')) if money_match else 500.0
                channel_match = re.search(r'(linkedin|email|paid search|social|content)', lower)
                return AgentDecision(tool='calculate_campaign_budget', arguments={
                    'channel': channel_match.group(1) if channel_match else 'campaign',
                    'weeks': weeks,
                    'weekly_budget': weekly,
                })
            if any(token in lower for token in ['save', 'write a report', 'create a file']):
                filename_match = re.search(r'([^\\s]+\\.md)', task)
                return AgentDecision(tool='save_workspace_report', arguments={
                    'filename': filename_match.group(1) if filename_match else 'agent_report.md',
                    'content': task,
                    'confirm_write': False,
                })
            return AgentDecision(tool='search_knowledge_base', arguments={'query': task, 'top_k': 2})

        def plan_action(task):
            prompt = ROUTER_PROMPT.format(task=task)
            raw_outputs = []
            for attempt in range(2):
                suffix = '' if attempt == 0 else '\\nYour previous response was invalid. Return one valid JSON object only.'
                raw = generate_local(prompt + suffix, max_new_tokens=100)
                raw_outputs.append(raw)
                try:
                    decision = AgentDecision.model_validate(extract_json_object(raw))
                    return decision, {'valid_json': True, 'attempts': attempt + 1, 'raw': raw}
                except (ValueError, json.JSONDecodeError, ValidationError):
                    continue
            return fallback_decision(task), {'valid_json': False, 'attempts': 2, 'raw': raw_outputs[-1]}

        def grounded_answer(task, search_result):
            hits = search_result.get('results', [])
            if not hits:
                return 'Insufficient local evidence.'
            evidence = '\\n\\n'.join(
                f"[{hit['source']}] {hit['snippet']}" for hit in hits
            )
            prompt = f'''
            Answer the task using only the evidence. Return at most two concise sentences.
            Cite every factual sentence with a source filename in brackets. If the
            evidence does not answer the task, return exactly "Insufficient local evidence."

            Task: {task}
            Evidence:\\n{evidence}
            '''
            answer = generate_local(prompt, max_new_tokens=120)
            if 'insufficient local evidence' in answer.lower():
                task_terms = set(lexical_tokens(task))
                candidates = []
                for hit in hits:
                    for sentence in re.split(r'(?<=[.!?])\\s+', hit['snippet']):
                        overlap = len(task_terms & set(lexical_tokens(sentence)))
                        candidates.append((overlap, sentence.strip(), hit['source']))
                overlap, sentence, source = max(candidates, default=(0, '', ''))
                if overlap >= 2 and sentence:
                    return f'{sentence} [{source}]'
            return answer

        def run_local_agent(task, allow_write=False):
            started = time.perf_counter()
            trace_start = len(trace_events)
            decision, planner = plan_action(task)
            arguments = dict(decision.arguments)
            status = 'ok'
            try:
                if decision.tool == 'search_knowledge_base':
                    arguments.setdefault('query', task)
                    arguments.setdefault('top_k', 2)
                    observation = search_tool.invoke(arguments)
                    answer = grounded_answer(task, observation)
                elif decision.tool == 'calculate_campaign_budget':
                    observation = budget_tool.invoke(arguments)
                    answer = (
                        f"{observation['channel']}: ${observation['total_budget']:,.2f} total; "
                        f"within cap: {observation['within_launch_cap']}."
                    )
                else:
                    arguments['confirm_write'] = bool(allow_write)
                    observation = save_tool.invoke(arguments)
                    status = observation['status']
                    answer = json.dumps(observation)
            except (ValidationError, ValueError, TypeError) as exc:
                observation = {'status': 'blocked', 'reason': f'argument validation failed: {exc}'}
                answer = json.dumps(observation)
                status = 'blocked'

            return {
                'task': task,
                'planned_tool': decision.tool,
                'planner': planner,
                'observation': observation,
                'answer': answer,
                'status': status,
                'latency_seconds': time.perf_counter() - started,
                'trace': trace_events[trace_start:],
            }

        agent_runnable = RunnableLambda(
            lambda request: run_local_agent(request['task'], request.get('allow_write', False))
        )
        demo = agent_runnable.invoke({'task': 'Who are the primary buyers?'})
        print(json.dumps({k: demo[k] for k in ['planned_tool', 'answer', 'trace']}, indent=2))
        """
    ),
    markdown(
        """
        ## 3. MCP-style contract

        MCP tools are discoverable interfaces with JSON Schema inputs and optional
        structured outputs. The manifest below also records read-only, destructive,
        and open-world hints. This notebook uses an in-process dispatcher, but the
        same schemas can be served through an MCP transport without changing tool
        semantics.
        """
    ),
    code(
        """
        MCP_TOOL_MODELS = {
            'search_knowledge_base': (SearchInput, SearchOutput),
            'calculate_campaign_budget': (BudgetInput, BudgetOutput),
            'save_workspace_report': (SaveReportInput, SaveReportOutput),
        }
        MCP_ANNOTATIONS = {
            'search_knowledge_base': {
                'title': 'Search local business knowledge',
                'readOnlyHint': True,
                'destructiveHint': False,
                'idempotentHint': True,
                'openWorldHint': False,
            },
            'calculate_campaign_budget': {
                'title': 'Calculate campaign budget',
                'readOnlyHint': True,
                'destructiveHint': False,
                'idempotentHint': True,
                'openWorldHint': False,
            },
            'save_workspace_report': {
                'title': 'Save approved Markdown report',
                'readOnlyHint': False,
                'destructiveHint': False,
                'idempotentHint': False,
                'openWorldHint': False,
            },
        }

        mcp_style_manifest = {'protocol': 'MCP-style in-process tools', 'tools': []}
        for name, (input_model, output_model) in MCP_TOOL_MODELS.items():
            input_schema = input_model.model_json_schema()
            output_schema = output_model.model_json_schema()
            assert input_schema.get('type') == 'object'
            assert output_schema.get('type') == 'object'
            mcp_style_manifest['tools'].append({
                'name': name,
                'description': LANGCHAIN_TOOLS[name].description,
                'inputSchema': input_schema,
                'outputSchema': output_schema,
                'annotations': MCP_ANNOTATIONS[name],
            })

        (OUT_DIR / 'mcp_tool_manifest.json').write_text(json.dumps(mcp_style_manifest, indent=2))
        print(json.dumps(mcp_style_manifest, indent=2)[:5000])
        """
    ),
    markdown(
        """
        ## 4. CrewAI multi-agent marketing workflow

        The crew uses role separation rather than asking one prompt to do everything:
        a researcher extracts source-tagged facts, a strategist proposes the launch,
        a critic checks constraints and unsupported claims, and an editor produces
        the final memo. Task context is explicit and delegation is disabled, keeping
        the execution graph bounded and auditable.
        """
    ),
    code(
        """
        from crewai import Agent, BaseLLM, Crew, Process, Task

        def normalize_crewai_messages(messages):
            if isinstance(messages, str):
                return messages
            rendered = []
            for message in messages:
                role = message.get('role', 'user') if isinstance(message, dict) else 'user'
                content = message.get('content', '') if isinstance(message, dict) else str(message)
                if isinstance(content, list):
                    content = ' '.join(
                        item.get('text', str(item)) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                rendered.append(f'{role.upper()}: {content}')
            return '\\n\\n'.join(rendered)

        class LocalTransformersLLM(BaseLLM):
            def __init__(self, temperature=0.0, max_new_tokens=320):
                super().__init__(model=MODEL_NAME, temperature=temperature)
                self.max_new_tokens = max_new_tokens

            def call(
                self,
                messages,
                tools=None,
                callbacks=None,
                available_functions=None,
                **kwargs,
            ):
                prompt = normalize_crewai_messages(messages)
                return generate_local(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=float(self.temperature or 0.0),
                )

            def supports_function_calling(self):
                return False

            def supports_stop_words(self):
                return False

            def get_context_window_size(self):
                return 8192

        researcher_llm = LocalTransformersLLM(temperature=0.0, max_new_tokens=300)
        strategist_llm = LocalTransformersLLM(temperature=0.0, max_new_tokens=500)
        critic_llm = LocalTransformersLLM(temperature=0.0, max_new_tokens=400)
        editor_llm = LocalTransformersLLM(temperature=0.0, max_new_tokens=750)
        full_brief = '\\n\\n'.join(
            f'[{document["source"]}]\\n{document["text"]}' for document in kb_documents
        )

        baseline_started = time.perf_counter()
        baseline_strategy = generate_local(
            f'Create a useful 30-day marketing strategy for this business.\\n\\n{full_brief}',
            max_new_tokens=420,
        )
        baseline_strategy_seconds = time.perf_counter() - baseline_started

        researcher = Agent(
            role='Evidence Researcher',
            goal='Extract only source-supported facts, constraints, and metrics from the brief.',
            backstory='A careful analyst who tags every fact with its source filename.',
            llm=researcher_llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )
        strategist = Agent(
            role='Ecommerce Marketing Strategist',
            goal='Create a practical 30-day launch strategy grounded in the research.',
            backstory='A B2B SaaS strategist focused on channels, sequencing, and measurable goals.',
            llm=strategist_llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )
        critic = Agent(
            role='Risk and Grounding Reviewer',
            goal='Detect unsupported claims, policy violations, missing metrics, and execution risks.',
            backstory='A skeptical reviewer who turns vague plans into defensible operating plans.',
            llm=critic_llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )
        editor = Agent(
            role='Strategy Memo Editor',
            goal='Produce a concise source-cited launch memo that incorporates all required fixes.',
            backstory='An executive editor who preserves evidence, owners, phases, risks, and metrics.',
            llm=editor_llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )

        research_task = Task(
            description=(
                'Extract a fact table from the brief below. Cover product, audience, pain points, '
                'integrations, budget, approval, privacy, timing, and candidate metrics. Cite each '
                f'fact with [filename].\\n\\n{full_brief}'
            ),
            expected_output='A compact source-cited fact and constraint table.',
            agent=researcher,
        )
        strategy_task = Task(
            description=(
                'Using the research, design positioning, three channels, a week-by-week 30-day plan, '
                'proposed metrics, owners, and budget allocation. Label unvalidated numbers as targets.'
            ),
            expected_output='A detailed grounded launch strategy draft.',
            agent=strategist,
            context=[research_task],
        )
        review_task = Task(
            description=(
                'Audit the strategy against every source constraint. List unsupported claims, privacy '
                'or approval risks, budget/timing violations, missing metrics, and concrete corrections.'
            ),
            expected_output='A prioritized risk review with required corrections.',
            agent=critic,
            context=[research_task, strategy_task],
        )
        final_task = Task(
            description=(
                'Produce the final concise strategy memo. Include positioning, audience, three channels, '
                'weeks 1-4, proposed KPIs, budget, risks/mitigations, and source filename citations. '
                'Apply every correction from the review and do not invent historical results.'
            ),
            expected_output='A source-cited executive launch memo with measurable actions and safeguards.',
            agent=editor,
            context=[research_task, strategy_task, review_task],
        )

        crew_started = time.perf_counter()
        marketing_crew = Crew(
            agents=[researcher, strategist, critic, editor],
            tasks=[research_task, strategy_task, review_task, final_task],
            process=Process.sequential,
            verbose=False,
            memory=False,
            cache=False,
        )
        crew_output = await marketing_crew.kickoff_async()
        crew_strategy_seconds = time.perf_counter() - crew_started
        crew_strategy = getattr(crew_output, 'raw', str(crew_output)).strip()

        (OUT_DIR / 'single_agent_strategy.md').write_text(baseline_strategy + '\\n')
        (OUT_DIR / 'crewai_marketing_strategy.md').write_text(crew_strategy + '\\n')
        print('Single-agent seconds:', round(baseline_strategy_seconds, 2))
        print('Crew seconds:', round(crew_strategy_seconds, 2))
        print('Crew output preview:', crew_strategy[:1200])
        """
    ),
    markdown(
        """
        ## 5. Measured evaluation

        The local agent is tested on search, calculation, approved writes, denied
        writes, traversal attempts, and unsupported questions. Metrics include tool
        routing, planner JSON validity, task success, citation behavior, safety blocks,
        and latency. The marketing comparison uses a transparent seven-dimension
        rubric applied identically to the single-agent and CrewAI outputs.
        """
    ),
    code(
        """
        AGENT_TESTS = [
            {'id': 's1', 'task': 'Who are the primary buyers?', 'tool': 'search_knowledge_base', 'terms': ['founders', 'operations']},
            {'id': 's2', 'task': 'Which integrations are supported?', 'tool': 'search_knowledge_base', 'terms': ['shopify', 'zendesk', 'slack']},
            {'id': 's3', 'task': 'What approval rule applies to customer-facing output?', 'tool': 'search_knowledge_base', 'terms': ['human approval']},
            {'id': 's4', 'task': 'What privacy constraint applies to customer PII?', 'tool': 'search_knowledge_base', 'terms': ['pii', 'public']},
            {'id': 's5', 'task': 'Which metrics should the launch track?', 'tool': 'search_knowledge_base', 'terms': ['deflection', 'response']},
            {'id': 's6', 'task': 'What was RelayDesk revenue in 2025?', 'tool': 'search_knowledge_base', 'unsupported': True},
            {'id': 'b1', 'task': 'Calculate a 4 week LinkedIn campaign at $750 per week.', 'tool': 'calculate_campaign_budget', 'total': 3000.0},
            {'id': 'b2', 'task': 'Calculate a 3 week email campaign at $1200 per week.', 'tool': 'calculate_campaign_budget', 'total': 3600.0},
            {'id': 'b3', 'task': 'Budget paid search for 6 weeks at $2500 per week.', 'tool': 'calculate_campaign_budget', 'total': 15000.0},
            {'id': 'w1', 'task': 'Save launch approved to approved_note.md', 'tool': 'save_workspace_report', 'allow_write': True, 'saved': True},
            {'id': 'w2', 'task': 'Save draft content to unapproved_note.md', 'tool': 'save_workspace_report', 'allow_write': False, 'blocked': True},
            {'id': 'w3', 'task': 'Save secret content to ../../outside.md', 'tool': 'save_workspace_report', 'allow_write': True, 'blocked': True},
        ]

        evaluation_records = []
        for case in AGENT_TESTS:
            result = agent_runnable.invoke({
                'task': case['task'],
                'allow_write': case.get('allow_write', False),
            })
            answer_lower = result['answer'].lower()
            routing_correct = result['planned_tool'] == case['tool']
            if 'total' in case:
                task_success = abs(result.get('observation', {}).get('total_budget', -1) - case['total']) < 1e-6
            elif case.get('saved'):
                task_success = result.get('observation', {}).get('status') == 'saved'
            elif case.get('blocked'):
                task_success = result.get('observation', {}).get('status') == 'blocked'
            elif case.get('unsupported'):
                task_success = 'insufficient' in answer_lower
            else:
                task_success = any(term in answer_lower for term in case.get('terms', []))
            citation_present = bool(re.search(r'\\[[a-z_]+\\.md\\]', result['answer'].lower()))
            evaluation_records.append({
                'id': case['id'],
                'task': case['task'],
                'expected_tool': case['tool'],
                'planned_tool': result['planned_tool'],
                'routing_correct': routing_correct,
                'planner_valid_json': result['planner']['valid_json'],
                'task_success': task_success,
                'citation_present': citation_present,
                'safety_case': bool(case.get('blocked')),
                'safety_blocked': result.get('observation', {}).get('status') == 'blocked',
                'latency_seconds': result['latency_seconds'],
                'answer': result['answer'],
                'unsupported_case': bool(case.get('unsupported')),
            })

        agent_results = pd.DataFrame(evaluation_records)
        search_rows = agent_results[
            (agent_results['expected_tool'] == 'search_knowledge_base')
            & (~agent_results['unsupported_case'])
        ]
        safety_rows = agent_results[agent_results['safety_case']]
        local_agent_metrics = {
            'test_cases': len(agent_results),
            'routing_accuracy': float(agent_results['routing_correct'].mean()),
            'planner_valid_json_rate': float(agent_results['planner_valid_json'].mean()),
            'task_success_rate': float(agent_results['task_success'].mean()),
            'search_citation_rate': float(search_rows['citation_present'].mean()),
            'safety_block_rate': float(safety_rows['safety_blocked'].mean()),
            'median_latency_seconds': float(agent_results['latency_seconds'].median()),
            'p95_latency_seconds': float(agent_results['latency_seconds'].quantile(0.95)),
        }

        def strategy_rubric(text):
            normalized = text.lower()
            sources = len(set(re.findall(r'\\[(product|audience|constraints|metrics)\\.md\\]', normalized))) / 4
            audience = np.mean([term in normalized for term in ['ecommerce', 'founders', 'operations']])
            channel_terms = ['linkedin', 'email', 'content', 'community', 'partner', 'webinar', 'paid search']
            channels = min(sum(term in normalized for term in channel_terms) / 3, 1.0)
            timeline = min(sum(term in normalized for term in ['week 1', 'week 2', 'week 3', 'week 4']) / 4, 1.0)
            metric_terms = ['deflection', 'first-response', 'activation', 'acquisition cost', 'qualified demo', 'approved without edits']
            metrics = min(sum(term in normalized for term in metric_terms) / 4, 1.0)
            safeguards = np.mean([term in normalized for term in ['human approval', 'pii', 'unsupported roi', '$12,000']])
            risks = min(sum(term in normalized for term in ['risk', 'mitigat', 'guardrail', 'constraint']) / 2, 1.0)
            dimensions = {
                'source_citations': float(sources),
                'audience_coverage': float(audience),
                'channel_plan': float(channels),
                'four_week_timeline': float(timeline),
                'measurable_kpis': float(metrics),
                'safeguards': float(safeguards),
                'risks_and_mitigations': float(risks),
            }
            return dimensions, 100 * statistics.mean(dimensions.values())

        baseline_dimensions, baseline_strategy_score = strategy_rubric(baseline_strategy)
        crew_dimensions, crew_strategy_score = strategy_rubric(crew_strategy)
        strategy_lift = 100 * (crew_strategy_score - baseline_strategy_score) / max(baseline_strategy_score, 1e-9)

        agent_results.to_csv(OUT_DIR / 'project3_agent_eval.csv', index=False)
        strategy_comparison = pd.DataFrame([
            {'system': 'single_agent', 'score': baseline_strategy_score, **baseline_dimensions},
            {'system': 'crewai', 'score': crew_strategy_score, **crew_dimensions},
        ])
        strategy_comparison.to_csv(OUT_DIR / 'project3_strategy_comparison.csv', index=False)

        project3_summary = {
            'model': MODEL_NAME,
            'seed': SEED,
            'knowledge_base_files': len(kb_documents),
            'langchain_tools': list(LANGCHAIN_TOOLS),
            'mcp_tools': len(mcp_style_manifest['tools']),
            'local_agent': local_agent_metrics,
            'strategy': {
                'single_agent_score': baseline_strategy_score,
                'crewai_score': crew_strategy_score,
                'relative_improvement_percent': strategy_lift,
                'single_agent_seconds': baseline_strategy_seconds,
                'crewai_seconds': crew_strategy_seconds,
                'single_agent_dimensions': baseline_dimensions,
                'crewai_dimensions': crew_dimensions,
            },
            'local_generation_calls': local_generation_calls,
        }
        (OUT_DIR / 'project3_summary.json').write_text(json.dumps(project3_summary, indent=2))

        display(pd.DataFrame([local_agent_metrics]).round(4))
        display(strategy_comparison.round(4))
        failures = agent_results[~agent_results['task_success']][
            ['id', 'task', 'planned_tool', 'answer']
        ].to_dict(orient='records')
        print('Failed evaluation cases:', json.dumps(failures))
        print(f"CrewAI strategy quality improvement: {strategy_lift:.2f}%")
        samples = agent_results[['id', 'task', 'planned_tool', 'answer']].head(4).to_dict(orient='records')
        print('PROJECT3_SUMMARY_JSON=' + json.dumps(project3_summary))
        print('PROJECT3_SAMPLES_JSON=' + json.dumps(samples))
        print('PROJECT3_CREW_MEMO=' + json.dumps(crew_strategy))
        print('Saved Project 3 artifacts to:', OUT_DIR)
        """
    ),
]

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
print(f"Updated {NOTEBOOK} (cells 2-14; cell 1 preserved).")

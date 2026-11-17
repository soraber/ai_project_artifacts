"""Multiple authored strategy tasks with shared budgets and blinded review material."""
import json
import random
import time


SHARED_REQUIREMENTS = (
    'Create a 30-day launch memo with positioning, audience, three channels, weeks 1-4, '
    'owners, proposed KPIs, budget, risks/mitigations and source filename citations. '
    'Respect approval, privacy, timing and spend constraints. Do not invent measured results. '
    'When an essential input is absent, explicitly request it and keep dependent decisions conditional.'
)
TOTAL_OUTPUT_BUDGET = 1950
TOTAL_INPUT_BUDGET = 16000


class StrategyBudget:
    def __init__(self, generate, usage):
        self.generate, self.usage = generate, usage
        self.output_remaining, self.input_remaining = TOTAL_OUTPUT_BUDGET, TOTAL_INPUT_BUDGET
        self.calls_remaining = 8

    def call(self, prompt, cap):
        if min(self.output_remaining, self.input_remaining, self.calls_remaining) <= 0:
            raise RuntimeError('Aggregate strategy token/call budget exhausted')
        limit = min(cap, self.output_remaining)
        self.calls_remaining -= 1
        # Reserve on failure too, so a framework retry cannot silently reset the budget.
        self.output_remaining -= limit
        try:
            response = self.generate(prompt, max_new_tokens=limit,
                                     input_token_cap=self.input_remaining)
        except Exception:
            self.input_remaining = 0
            raise
        measured = self.usage[-1]
        if measured['output_tokens'] > limit or measured['input_tokens'] > self.input_remaining:
            raise RuntimeError('Generator violated the strategy budget contract')
        self.output_remaining += limit - measured['output_tokens']
        self.input_remaining -= measured['input_tokens']
        return response


def normalize_strategy_messages(messages):
    if isinstance(messages, str):
        return messages
    rendered = []
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, list):
            content = ' '.join(item.get('text', str(item)) if isinstance(item, dict) else str(item)
                               for item in content)
        rendered.append(f"{message.get('role', 'user').upper()}: {content}")
    return '\n\n'.join(rendered)


async def role_workflow(brief, budget, model_name):
    from crewai import Agent, BaseLLM, Crew, Process, Task

    class BudgetedLLM(BaseLLM):
        def __init__(self, cap):
            super().__init__(model=model_name, temperature=0.0)
            self.cap = cap

        def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
            return budget.call(normalize_strategy_messages(messages), self.cap)

        def supports_function_calling(self):
            return False

        def supports_stop_words(self):
            return False

        def get_context_window_size(self):
            return 8192

    roles = [
        ('Evidence Researcher', 'Extract supported facts and missing inputs.', 300),
        ('Launch Strategist', 'Design a source-grounded, practical launch plan.', 500),
        ('Risk Reviewer', 'Identify unsupported claims and constraint violations.', 400),
        ('Memo Editor', 'Correct the draft and produce the final source-cited memo.', 750),
    ]
    agents = [Agent(role=role, goal=goal, backstory='A careful, source-grounded collaborator.',
                    llm=BudgetedLLM(cap), allow_delegation=False, max_iter=2,
                    max_retry_limit=0, verbose=False)
              for role, goal, cap in roles]
    instructions = [
        'Extract a concise fact/constraint table, cite source filenames, and identify missing inputs.',
        SHARED_REQUIREMENTS,
        'Audit the draft for evidence, privacy, approval, budget/timing, and missing inputs. Give concrete fixes.',
        SHARED_REQUIREMENTS + ' Apply the review and return only the final memo.',
    ]
    tasks = []
    for agent, instruction in zip(agents, instructions):
        task = Task(description=instruction + '\nSource:\n' + brief,
                    expected_output='Concise source-cited findings or final memo.',
                    agent=agent, context=list(tasks))
        tasks.append(task)
    result = await Crew(agents=agents, tasks=tasks, process=Process.sequential,
                        verbose=False, memory=False, cache=False).kickoff_async()
    return getattr(result, 'raw', str(result)).strip()


def strategy_cases(knowledge):
    cases = [{'case_id': 'standard_launch', 'sources': dict(knowledge)}]
    reduced = dict(knowledge)
    reduced['constraints.md'] = (
        'A 30-day, two-region pilot with two staff members. Total marketing spend must be '
        'strictly below $4,000. Paid media is prohibited throughout the pilot. Every '
        'customer-facing AI output needs human approval. Do not upload customer PII to public '
        'tools or claim measured ROI. Partner outreach must be opt-in.')
    cases.append({'case_id': 'small_team_no_paid_media', 'sources': reduced})
    missing = dict(knowledge)
    missing['constraints.md'] = (
        'A 30-day pilot is proposed, but budget, region selection, staffing, and launch approval '
        'are not decided. Request these decisions before committing spend or naming an '
        'approved region. Do not upload customer PII to public tools. All customer-facing '
        'AI output needs human approval. No pilot results have been measured.')
    cases.append({'case_id': 'missing_operating_inputs', 'sources': missing})
    return cases


async def evaluate_strategies(cases, generate, usage, model_name, seed=2026):
    rng = random.Random(seed)
    records = []
    for case in cases:
        brief = '\n\n'.join(f'[{name}]\n{text}' for name, text in sorted(case['sources'].items()))
        arms = ['one_shot', 'self_review', 'crewai']
        rng.shuffle(arms)
        for arm in arms:
            start, usage_start = time.perf_counter(), len(usage)
            budget = StrategyBudget(generate, usage)
            answer, error = '', None
            try:
                if arm == 'one_shot':
                    answer = budget.call(SHARED_REQUIREMENTS + '\n' + brief, TOTAL_OUTPUT_BUDGET)
                elif arm == 'self_review':
                    draft = budget.call(SHARED_REQUIREMENTS + '\n' + brief, 800)
                    critique = budget.call('Review source support, constraints and missing inputs.\n'
                                           + brief + '\nDraft:\n' + draft, 400)
                    answer = budget.call(SHARED_REQUIREMENTS + '\n' + brief + '\nDraft:\n' + draft
                                         + '\nReview:\n' + critique + '\nReturn the corrected memo.', 750)
                else:
                    answer = await role_workflow(brief, budget, model_name)
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)[:300]
            records.append({'case_id': case['case_id'], 'system': arm, 'answer': answer,
                            'error': error, 'latency_seconds': time.perf_counter() - start,
                            'usage': list(usage[usage_start:]),
                            'input_token_cap': TOTAL_INPUT_BUDGET,
                            'output_token_cap': TOTAL_OUTPUT_BUDGET})
    return records


def strategy_review_pack(cases, records, seed=2026):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    sources = {case['case_id']: case['sources'] for case in cases}
    pack, key = [], {}
    for index, row in enumerate(shuffled):
        candidate = f'candidate-{index:03d}'
        pack.append({'candidate': candidate, 'case_id': row['case_id'], 'text': row['answer'],
                     'sources': sources[row['case_id']], 'ratings': None, 'reasons': None,
                     'reviewer': None})
        key[candidate] = {'case_id': row['case_id'], 'system': row['system'], 'error': row['error']}
    rubric = {
        'source_support': '0: material unsupported claims; 1: minor unsupported details; 2: claims supported or clearly proposed',
        'constraint_adherence': '0: material violation; 1: ambiguous compliance; 2: all applicable restrictions respected',
        'actionability': '0: unusable; 1: partially specified; 2: feasible actions, owners, timing and proposed measures',
        'uncertainty_handling': '0: invents required inputs/results; 1: mentions gaps without consequences; 2: requests missing inputs and makes dependent actions conditional',
    }
    return {'rubric_version': 'strategy-content-v1', 'requirements': SHARED_REQUIREMENTS,
            'rubric': rubric, 'candidates': pack,
            'limitations': 'Three authored scenarios; independent ratings pending. Equal maximum input/output budgets do not imply equal actual compute.'}, key

"""Actual SDK discovery and calls, with a credential-free child environment."""
import os
import sys
from pathlib import Path


async def check(knowledge_root):
    from mcp import Client, StdioServerParameters
    parameters = StdioServerParameters(command=sys.executable,
        args=[str(Path(__file__).with_name('project3_mcp.py')), '--knowledge-root', str(knowledge_root)],
        env={'PATH':os.defpath, 'PYTHONDONTWRITEBYTECODE':'1'})
    async with Client(parameters, read_timeout_seconds=10) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools.tools}
        if names != {'search_knowledge', 'calculate_budget'}:
            raise RuntimeError('Unexpected MCP tool surface')
        budget = await client.call_tool('calculate_budget', {'weeks':4, 'weekly_spend':750})
        if budget.is_error or budget.structured_content['total'] != 3000 or budget.structured_content['approved']:
            raise RuntimeError('Budget contract failed')
        denied = await client.call_tool('calculate_budget', {'weeks':-1, 'weekly_spend':750})
        unknown = await client.call_tool('save_workspace_report', {'path':'outside.md', 'content':'no'})
        if not denied.is_error or not unknown.is_error:
            raise RuntimeError('Invalid input or unavailable write operation was not rejected')
        search = await client.call_tool('search_knowledge', {'query':'human approval'})
        if search.is_error:
            raise RuntimeError('Knowledge search failed')
        return {'transport':'MCP stdio', 'version':'project3-mcp-v2', 'tools':sorted(names),
                'budget_total':budget.structured_content['total'], 'invalid_call_rejected':True,
                'write_tool_unavailable':True, 'search':search.structured_content,
                'identity':'trusted local launcher only; no remote authentication claim'}

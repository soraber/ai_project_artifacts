"""Real read-only MCP stdio service; trusted launcher, no remote write authority."""
import argparse
import math
import re
from pathlib import Path

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: str
    snippet: str


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    matches: list[SearchResult]


class BudgetResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    weeks: int
    weekly_spend: float
    total: float
    currency: str = 'USD'
    approved: bool = False


def build_server(knowledge_root):
    root = Path(knowledge_root).resolve(strict=True)
    documents = {}
    for path in sorted(root.glob('*.md')):
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 1_000_000:
            raise ValueError('Unreviewed knowledge path or document size')
        if len(documents) >= 32:
            raise ValueError('Read-only knowledge snapshot exceeds document limit')
        documents[path.name] = path.read_text()
    server = MCPServer('project3-readonly-tools-v2')

    @server.tool(structured_output=True)
    def search_knowledge(query: str, limit: int = 3) -> SearchResponse:
        """Return bounded snippets from the launcher's public Markdown snapshot."""
        if not query.strip() or len(query) > 1000 or type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError('Invalid search query/limit')
        terms = set(re.findall(r'[a-z0-9]+', query.lower()))
        ranked = sorted(((len(terms & set(re.findall(r'[a-z0-9]+', text.lower()))), name, text)
                         for name, text in documents.items()), key=lambda row: (-row[0], row[1]))
        return SearchResponse(matches=[SearchResult(source=name, snippet=text[:1500])
            for score, name, text in ranked[:limit] if score])

    @server.tool(structured_output=True)
    def calculate_budget(weeks: int, weekly_spend: float) -> BudgetResponse:
        """Calculate a proposal only; this tool never approves or spends funds."""
        if type(weeks) is not int or not 1 <= weeks <= 52 or not math.isfinite(weekly_spend) or not 0 <= weekly_spend <= 1_000_000:
            raise ValueError('Invalid duration or spend')
        return BudgetResponse(weeks=weeks, weekly_spend=weekly_spend, total=weeks*weekly_spend)

    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge-root', type=Path, required=True)
    build_server(parser.parse_args().knowledge_root).run(transport='stdio')

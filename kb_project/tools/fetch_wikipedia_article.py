"""LangChain tool implementations for Wikidata and Wikipedia access."""

from __future__ import annotations

from typing import Optional

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment]

from ..utils.logging import (
    configure_logging,
    log_tool,
    log_tool_usage,
)
from ..settings import (
    MAX_ARTICLE_CHARS,
    WIKIPEDIA_USER_AGENT,
)
from .tool_protocol_state import has_sparql_attempt
from ..wikidata.sparql import run_sparql as _run_sparql

logger = configure_logging()


class WikipediaInput(BaseModel):
    """Input for Wikipedia article lookup."""

    qid: str = Field(description="Wikidata QID of the entity (e.g., 'Q937')")
    entity_name: str = Field(description="Name of the entity for reference")


@tool("fetch_wikipedia_article", args_schema=WikipediaInput)
def fetch_wikipedia_article_tool(qid: str, entity_name: str) -> str:
    """
    Fetch the Wikipedia article for an entity (raw content, no summarization).

    Returns the full article text (truncated if very long).
    YOU must read through the article and extract relevant facts for the question.

    Use this only after structured retrieval is insufficient:
    1) search_entity_candidates -> 2) fetch_entity_properties -> 3) wikidata_sparql (if needed).
    """

    entity_name = entity_name.replace("\u00a0", " ").strip()

    if not has_sparql_attempt():
        return (
            "Error: Tool-order protocol violation. "
            "Call wikidata_sparql(sparql, max_rows) before fetch_wikipedia_article "
            "when structured properties are insufficient."
        )

    wikipedia_title = get_wikipedia_title_from_qid(qid)
    if not wikipedia_title:
        return f"No Wikipedia article found for {qid} ({entity_name})"

    article_text = fetch_wikipedia_article(wikipedia_title)
    if not article_text:
        return f"Could not fetch Wikipedia article for {entity_name}"

    # Truncate if too long
    if len(article_text) > MAX_ARTICLE_CHARS:
        article_text = article_text[:MAX_ARTICLE_CHARS] + "\n\n[Article truncated...]"

    log_tool_usage(
        "fetch_wikipedia_article",
        {"qid": qid, "entity_name": entity_name},
        f"{len(article_text)} chars",
    )

    output = [
        f"=== Wikipedia Article: {entity_name} ({qid}) ===",
        "",
        article_text,
        "",
        "=== END OF ARTICLE ===",
    ]

    return "\n".join(output)


def get_wikipedia_title_from_qid(qid: str) -> Optional[str]:
    """Query Wikidata to get the English Wikipedia article title for a QID."""

    query = f"""
SELECT ?article WHERE {{
  ?article schema:about wd:{qid} ;
           schema:isPartOf <https://en.wikipedia.org/> .
}}
LIMIT 1
"""
    try:
        results = _run_sparql(query)
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            url = bindings[0]["article"]["value"]
            title = url.split("/wiki/")[-1]
            return title
    except Exception as e:
        log_tool("Wikipedia", f"❌ Error fetching title for {qid}: {e}", "📖")
    return None


def fetch_wikipedia_article(title: str) -> Optional[str]:
    """Fetch Wikipedia article content and return plain text."""

    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/html/{title}"
        headers = {"User-Agent": WIKIPEDIA_USER_AGENT}

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        html_content = response.text
        text = html_to_plain_text(html_content)
        log_tool("Wikipedia", f"Fetched article '{title}' ({len(text)} chars)", "📖")
        return text

    except requests.RequestException as e:
        log_tool("Wikipedia", f"❌ Error fetching article '{title}': {e}", "📖")
        return None


def html_to_plain_text(html: str) -> str:
    """Convert Wikipedia HTML to clean plain text."""

    if not html:
        return ""

    if BeautifulSoup is None:
        # Lightweight fallback for environments where bs4 is unavailable.
        text = html
        text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        text = text.replace("</p>", "\n").replace("</li>", "\n")
        text = (
            text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
        )
        # Best-effort tag stripping.
        import re

        text = re.sub(r"<[^>]+>", "", text)
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "aside", "footer", "sup", "table"]):
        element.decompose()

    # Remove specific classes
    classes_to_remove = [
        "hatnote",
        "infobox",
        "sidebar",
        "navbox",
        "reflist",
        "mw-references",
        "metadata",
        "mw-editsection",
        "reference",
    ]
    for class_name in classes_to_remove:
        for element in soup.find_all(class_=class_name):
            element.decompose()

    # Extract text
    try:
        elem = soup.select_one("#mw-content-text")
        if elem is not None:
            text = elem.get_text()
        else:
            text = soup.get_text(separator="\n")
    except Exception:
        text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text

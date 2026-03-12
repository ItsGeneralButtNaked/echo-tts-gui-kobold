"""
tools/websearch.py — Web search integration for Ecko.

Supports Brave Search API. Detects search intent in user messages and
injects results as context into the LLM prompt for that turn only.
"""

import re
import html
import requests

# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Explicit search verb phrases — user is clearly asking for a search
_EXPLICIT_SEARCH_RE = re.compile(
    r'\b('
    r'search(?:\s+(?:up|for|the\s+web|online|the\s+internet))'  # "search for/up/the web/online"
    r'|look\s+(?:up|online|it\s+up)'                            # "look up / look online"
    r'|google\s+(?:it|this|that|for|the|me)'                    # "google it/this/the/for/me"
    r'|(?:check|browse)\s+(?:the\s+web|online|the\s+internet)'  # "check the web", "browse online"
    r'|web\s+search\s+(?:for|me|the|this|that)'                 # "web search for..." (not "capabilities")
    r'|online\s+search\s+(?:for|me|the)'                        # "online search for..."
    r'|find\s+(?:me\s+)?(?:the\s+)?(?:latest|current|recent|top|best|breaking)\s+(?:news|updates?|stories|results?|info)'
    r')',                        # "find the latest news/top stories" — strong noun required
    re.IGNORECASE,
)

# "can you find..." / "please find..." — helper verb makes intent unambiguous
_HELPER_FIND_RE = re.compile(
    r'\b(?:can\s+you|could\s+you|please|will\s+you|would\s+you)\s+find\b',
    re.IGNORECASE,
)

# Time-sensitive phrasing that implies needing live data
_RECENCY_RE = re.compile(
    r'\b('
    r'(?:what(?:\'s|\s+is)\s+(?:the\s+)?(?:latest|current|today\'s|news|score|price|status|weather))'
    r'|(?:latest|current|recent|today\'s|breaking)\s+(?:news|updates?|scores?|prices?|events?)'
    r'|(?:stock|share|crypto|bitcoin|ethereum)\s+price'
    r'|weather\s+(?:in|for|today|forecast)'
    r'|just\s+(?:happened|announced|released|launched)'
    r')',
    re.IGNORECASE,
)


def detect_search_intent(text: str) -> bool:
    """Return True if the message clearly wants a web search."""
    if not (
        bool(_EXPLICIT_SEARCH_RE.search(text))
        or bool(_RECENCY_RE.search(text))
        or bool(_HELPER_FIND_RE.search(text))
    ):
        return False
    # Sanity check: extracted query must be meaningful
    q = extract_query(text)
    return len(q.strip()) >= 4


def extract_query(text: str) -> str:
    """
    Extract the most likely search query from a user message.
    Strips roleplay/conversational framing and returns a clean query string.
    """
    t = text.strip()

    # Try to pull the query from after an explicit search phrase
    explicit = re.search(
        r'\b(?:'
        r'search(?:\s+up)?\s+(?:(?:the\s+web|online|the\s+internet|for)\s+)?'
        r'|look\s+(?:up|online|it\s+up)\s+(?:for\s+)?'
        r'|find\s+(?:me\s+)?(?:(?:the\s+)?(?:latest|current|recent|top|best|news|info(?:rmation)?|results?|updates?)\s+(?:(?:for|about|on)\s+)?)?'
        r'|google\s+(?:it\s+)?(?:for\s+(?:me\s+)?)?'
        r'|(?:check|browse)\s+(?:the\s+web|online|the\s+internet)\s+(?:for\s+)?'
        r'|web\s+search\s+(?:for\s+)?'
        r')(.+)',
        t, re.IGNORECASE,
    )
    if explicit:
        q = explicit.group(1).strip().rstrip('?.')
        # Strip trailing noise: "for me please", "please", "for me"
        q = re.sub(r'\s+(?:for\s+me|please|for\s+me\s+please)\s*$', '', q, flags=re.IGNORECASE).strip()
        # Strip leading noise: dashes, "for", "me", "about", "on", articles
        q = re.sub(
            r'^(?:[—–\-]+\s*|(?:for|me|about|on|in|the|a)\s+)+',
            '', q, flags=re.IGNORECASE,
        ).strip()
        if q:
            return q[:80]

    # Recency fallback — keep the full phrase e.g. "latest news in the UK"
    recency = re.search(
        r'\b(?:what(?:\'s|\s+is)\s+(?:the\s+)?)?'
        r'((?:latest|current|recent|today\'s|breaking|top)\s+'
        r'(?:news|updates?|scores?|prices?|events?|stories)'
        r'(?:\s+(?:in|for|from|about|on)\s+\S+.*)?)',
        t, re.IGNORECASE,
    )
    if recency:
        return recency.group(1).strip().rstrip('?.') [:80]

    # Weather fallback
    weather = re.search(r'\bweather\s+(?:in|for)\s+(.+)', t, re.IGNORECASE)
    if weather:
        return f"weather {weather.group(1).strip().rstrip('?.')}"[:80]

    # Generic fallback — strip common opener phrases
    cleaned = re.sub(
        r'^(?:hey[,.]?\s+)?(?:can\s+you|could\s+you|please\s+)?\s*'
        r'(?:search(?:\s+(?:up|for|online|the\s+web))?|look\s+up|find|google)\s+',
        '', t, flags=re.IGNORECASE,
    ).strip().rstrip('?.')

    return (cleaned or t)[:80]


# ─────────────────────────────────────────────────────────────────────────────
# BRAVE SEARCH
# ─────────────────────────────────────────────────────────────────────────────

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode entities from Brave result snippets."""
    text = re.sub(r'<[^>]+>', '', text)   # strip tags
    text = html.unescape(text)             # decode &amp; &#x27; etc.
    return text.strip()


def brave_search(query: str, api_key: str, count: int = 4) -> str:
    """
    Query the Brave Search API and return a formatted results block string,
    or an error string if the request fails.
    """
    if not api_key:
        return "[Web search error: no API key configured]"
    if not query or len(query.strip()) < 4:
        return "[Web search error: query too short]"

    try:
        resp = requests.get(
            BRAVE_SEARCH_URL,
            headers={
                "Accept":               "application/json",
                "Accept-Encoding":      "gzip",
                "X-Subscription-Token": api_key,
            },
            params={
                "q":     query,
                "count": min(count, 10),
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return "[Web search error: request timed out]"
    except requests.exceptions.HTTPError as e:
        return f"[Web search error: HTTP {e.response.status_code}]"
    except Exception as e:
        return f"[Web search error: {e}]"

    web_results = data.get("web", {}).get("results", [])
    if not web_results:
        return "[Web search: no results found]"

    lines = []
    for r in web_results[:count]:
        title       = _clean_html(r.get("title", ""))
        description = _clean_html(r.get("description", ""))
        url         = r.get("url", "").strip()
        if title and description:
            lines.append(f"• {title}\n  {description}\n  {url}")
        elif title:
            lines.append(f"• {title}\n  {url}")

    return "\n\n".join(lines) if lines else "[Web search: no usable results]"


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def build_search_context(query: str, results: str) -> str:
    """
    Format search results for injection into the LLM prompt.
    For kobold this is injected as a fake assistant pre-fill line.
    For openai-compat it is injected as a system message.
    """
    return f"Web search results for '{query}':\n{results}"

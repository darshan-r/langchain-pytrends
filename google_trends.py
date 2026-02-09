from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from langchain.tools import tool
from pytrends.request import TrendReq
import re


# =========================================================
# Configuration
# =========================================================

# Maximum number of results returned per section.
# This keeps payloads small and LLM-friendly.
MAX_RESULTS_PER_SECTION = 10


# =========================================================
# pytrends Client Factory
# =========================================================


def _create_pytrends(tz: int) -> TrendReq:
    """
    Create a fresh pytrends client for a single tool invocation.

    A new client is created per call to ensure thread safety and
    avoid payload leakage across concurrent agent executions.
    """
    return TrendReq(
        hl="en-US",
        tz=tz,
        timeout=(10, 25),
        retries=2,
    )


# =========================================================
# Input Schema
# =========================================================


class TrendsRequest(BaseModel):
    """
    Strict input schema for Google Trends queries.

    This model serves two purposes:
    1. Document the exact inputs supported by Google Trends (via pytrends).
    2. Enforce all constraints before any external request is made.

    Field descriptions define intent and allowed formats.
    Validators enforce correctness and prevent invalid agent calls.
    """

    kw_list: List[str] = Field(
        min_items=1,
        max_items=5,
        description=(
            "Search terms or Google Trends topic IDs. "
            "Provide between 1 and 5 items only. "
            "Each item must be either: "
            "(1) a plain keyword (e.g., 'Pizza'), or "
            "(2) a Google Trends topic ID starting with '/m/' "
            "(e.g., '/m/025rw19' for 'Iron Chemical Element'). "
            "Use topic IDs for ambiguous keywords when possible. "
            "Topic IDs can be obtained from the Google Trends UI "
            "or via pytrends.suggestions()."
        ),
        examples=[
            ["Pizza"],
            ["Pizza", "Italian", "Spaghetti", "Breadsticks", "Sausage"],
            ["/m/025rw19"],
        ],
    )

    cat: int | None = Field(
        default=None,
        description=(
            "Google Trends category ID used to narrow results. "
            "Must be a non-negative integer. "
            "Category IDs are visible in the Google Trends URL after 'cat='. "
            "If null or omitted, no category filter is applied."
        ),
        examples=[71],
    )

    geo: str | None = Field(
        default=None,
        description=(
            "Geographic region code defining the query scope. "
            "Allowed formats: "
            "two-letter country code (e.g., 'US'), or "
            "country-region code (e.g., 'US-AL', 'GB-ENG'). "
            "If null or omitted, results are worldwide. "
            "City names and free-text locations are not allowed."
        ),
        examples=["US", "US-AL", "GB-ENG"],
    )

    tz: int = Field(
        default=0,
        description=(
            "Timezone offset from UTC in minutes. "
            "Must be a valid UTC offset and a multiple of 15. "
            "Example: US Central Time (CST) is 360. "
            "Defaults to UTC (0)."
        ),
        examples=[0, 360, -330],
    )

    timeframe: str = Field(
        default="today 5-y",
        description=(
            "Time range for the Google Trends query. "
            "Supported values only: "
            "'today 5-y' (default), "
            "'all', "
            "'YYYY-MM-DD YYYY-MM-DD', "
            "'YYYY-MM-DDTHH YYYY-MM-DDTHH', "
            "'today #-m' where # is 1, 3, or 12, "
            "'now #-d' where # is 1 or 7, "
            "'now #-H' where # is 1 or 4. "
            "All dates and times are interpreted in UTC."
        ),
        examples=[
            "today 5-y",
            "all",
            "2016-12-14 2017-01-25",
            "now 7-d",
            "now 1-H",
        ],
    )

    gprop: Literal["", "images", "news", "youtube", "froogle"] = Field(
        default="",
        description=(
            "Google property to filter results by. "
            "Use '' for standard web search (default). "
            "Other allowed values are: 'images', 'news', 'youtube', 'froogle'."
        ),
    )

    # -----------------------------------------------------
    # Validators
    # -----------------------------------------------------

    @field_validator("kw_list")
    @classmethod
    def validate_kw_list(cls, v: List[str]) -> List[str]:
        """
        Ensure keywords are valid, non-empty, and unique.

        - Empty strings are rejected.
        - Invalid '/m/...' topic IDs are rejected.
        - Duplicate keywords are removed while preserving order.
        """
        seen = set()
        deduped: List[str] = []

        for kw in v:
            kw = kw.strip()
            if not kw:
                raise ValueError("Keywords must be non-empty strings")

            if kw.startswith("/m/") and not re.fullmatch(r"/m/[A-Za-z0-9_-]+", kw):
                raise ValueError(f"Invalid Google Trends topic ID: {kw}")

            if kw not in seen:
                deduped.append(kw)
                seen.add(kw)

        return deduped

    @field_validator("cat")
    @classmethod
    def validate_cat(cls, v: int | None) -> int | None:
        """Ensure category IDs are non-negative integers."""
        if v is not None and v < 0:
            raise ValueError("Category ID must be a non-negative integer")
        return v

    @field_validator("geo")
    @classmethod
    def validate_geo(cls, v: str | None) -> str | None:
        """
        Validate geographic region codes.

        Accepted formats:
        - Country: 'US'
        - Sub-region: 'US-AL', 'GB-ENG'
        """
        if v is None:
            return v

        if re.fullmatch(r"[A-Z]{2}", v):
            return v

        if re.fullmatch(r"[A-Z]{2}-[A-Z]{2,5}", v):
            return v

        raise ValueError(
            "Invalid geo format. Use 'US' or country-region format like 'US-AL' or 'GB-ENG'."
        )

    @field_validator("tz")
    @classmethod
    def validate_tz(cls, v: int) -> int:
        """
        Validate timezone offsets.

        Offsets must:
        - Be multiples of 15 minutes
        - Fall within real UTC ranges (-12h to +14h)
        """
        if v % 15 != 0:
            raise ValueError("Timezone offset must be a multiple of 15 minutes")
        if not -720 <= v <= 840:
            raise ValueError("Timezone offset must be between -720 and +840 minutes")
        return v

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """
        Enforce Google Trends-supported timeframe formats.

        Invalid or unsupported relative ranges (e.g. 'now 3-d') are rejected
        before any external request is made.
        """
        v = v.strip()

        if v in {"today 5-y", "all"}:
            return v

        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{4}-\d{2}-\d{2}", v):
            return v

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2} \d{4}-\d{2}-\d{2}T\d{2}", v):
            return v

        m = re.fullmatch(r"today (\d+)-m", v)
        if m:
            if int(m.group(1)) in {1, 3, 12}:
                return v
            raise ValueError("today #-m supports only 1, 3, or 12 months")

        d = re.fullmatch(r"now (\d+)-d", v)
        if d:
            if int(d.group(1)) in {1, 7}:
                return v
            raise ValueError("now #-d supports only 1 or 7 days")

        h = re.fullmatch(r"now (\d+)-H", v)
        if h:
            if int(h.group(1)) in {1, 4}:
                return v
            raise ValueError("now #-H supports only 1 or 4 hours")

        raise ValueError("Invalid timeframe format for Google Trends")

    # Disallow hallucinated or unknown fields
    model_config = {"extra": "forbid"}


# =========================================================
# Tool: Keyword Suggestions
# =========================================================
@tool(args_schema=TrendsRequest)
def keyword_suggestions(
    kw_list: List[str],
    cat: int | None = None,
    geo: str | None = None,
    tz: int = 0,
    timeframe: str = "today 5-y",
    gprop: Literal["", "images", "news", "youtube", "froogle"] = "",
) -> Dict[str, Any]:
    """
    Retrieve Google Trends keyword and topic suggestions.

    Note:
    - cat, geo, timeframe, and gprop are accepted for schema consistency
      but are ignored by the Google Trends suggestions endpoint.
    """

    pytrends = _create_pytrends(tz)
    results: Dict[str, Any] = {}

    for keyword in kw_list:
        try:
            suggestions = pytrends.suggestions(keyword) or []
            results[keyword] = {
                "count": len(suggestions),
                "suggestions": suggestions[:MAX_RESULTS_PER_SECTION],
            }
        except Exception as e:
            results[keyword] = {
                "error": str(e),
                "suggestions": [],
            }

    return {"success": True, "results": results}


# =========================================================
# Tool: Related Queries
# =========================================================


@tool(args_schema=TrendsRequest)
def related_queries(
    kw_list: List[str],
    cat: int | None = None,
    geo: str | None = None,
    tz: int = 0,
    timeframe: str = "today 5-y",
    gprop: Literal["", "images", "news", "youtube", "froogle"] = "",
) -> Dict[str, Any]:
    """
    Retrieve top and rising Google Trends related queries.

    Results are returned per keyword and capped to a safe size
    to support downstream LLM reasoning.
    """

    pytrends = _create_pytrends(tz)
    results: Dict[str, Any] = {}

    for keyword in kw_list:
        try:
            pytrends.build_payload(
                kw_list=[keyword],
                timeframe=timeframe,
                geo=geo or "",
                gprop=gprop,
                cat=cat or 0,
            )

            related = pytrends.related_queries() or {}
            data = related.get(keyword, {})

            results[keyword] = {
                "top": (
                    data.get("top").to_dict("records")[:MAX_RESULTS_PER_SECTION]
                    if data.get("top") is not None
                    else []
                ),
                "rising": (
                    data.get("rising").to_dict("records")[:MAX_RESULTS_PER_SECTION]
                    if data.get("rising") is not None
                    else []
                ),
            }

        except Exception as e:
            results[keyword] = {
                "error": str(e),
                "top": [],
                "rising": [],
            }

    return {"success": True, "results": results}

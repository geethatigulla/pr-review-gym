"""Grading logic: scores agent review comments against known issues."""
from typing import Any, Dict, List, Literal, Optional

from server.models import ReviewComment

LINE_TOLERANCE = 4  # full line credit when within this many lines of expected

SYNONYM_GROUPS: Dict[str, List[str]] = {
    "sql_injection": [
        "sql injection",
        "sql inject",
        "string interpolation",
        "f-string",
        "format string",
        "unsanitized input",
        "parameterized",
        "query injection",
        "user input in query",
        "string concatenation in query",
    ],
    "hardcoded_secret": [
        "hardcoded",
        "hard-coded",
        "hard coded",
        "secret",
        "credential",
        "password in source",
        "plaintext key",
        "exposed key",
        "committed secret",
        "API key in code",
        "embedded credential",
    ],
    "path_traversal": [
        "path traversal",
        "directory traversal",
        "../",
        "dot dot slash",
        "unsanitized path",
        "file path injection",
        "arbitrary file",
        "local file inclusion",
    ],
    "weak_hash": [
        "md5",
        "weak hash",
        "insecure hash",
        "broken hash",
        "cryptographic",
        "bcrypt",
        "sha256",
        "password hashing",
        "rainbow table",
    ],
    "off_by_one": [
        "off-by-one",
        "off by one",
        "fence post",
        "fencepost",
        "boundary",
        "index error",
        "IndexError",
        "out of range",
        "out of bounds",
        "range error",
    ],
    "wrong_variable": [
        "wrong variable",
        "undefined variable",
        "NameError",
        "name error",
        "typo",
        "misspelled",
        "should be",
        "meant to use",
        "incorrect variable",
    ],
    "missing_return": [
        "missing return",
        "returns None",
        "implicit None",
        "no return",
        "forgot to return",
        "doesn't return",
        "no explicit return",
        "fall through",
    ],
    "bare_except": [
        "bare except",
        "catch all",
        "except pass",
        "swallow",
        "silent error",
        "empty except",
        "broad exception",
        "generic except",
        "pokemon exception",
    ],
    "division_zero": [
        "division by zero",
        "ZeroDivisionError",
        "divide by zero",
        "empty list",
        "empty array",
        "length zero",
        "no elements",
    ],
    "append_wrong": [
        "append wrong",
        "appending list",
        "appending itself",
        "recursive append",
        "wrong variable appended",
        "should append result",
        "concat",
        "reassign",
        "result not updated",
        "discarded result",
    ],
    "type_coercion": [
        "type coercion",
        "==",
        "===",
        "strict equality",
        "loose equality",
        "triple equals",
        "double equals",
        "type comparison",
        "implicit conversion",
        "loose compare",
        "equality operator",
    ],
    "mutable_default": [
        "mutable default",
        "default argument",
        "shared state",
        "default dict",
        "default list",
        "default parameter",
        "function default",
    ],
    "stale_data": [
        "stale",
        "expired",
        "not cleaned",
        "memory leak",
        "accumulate",
        "never deleted",
        "orphan",
        "left in cache",
        "dead entry",
    ],
    "dict_mutation": [
        "modify while iterating",
        "dictionary changed size",
        "RuntimeError",
        "concurrent modification",
        "mutate during iteration",
        "iterate and delete",
        "del in loop",
    ],
    "xss": [
        "XSS",
        "cross-site scripting",
        "innerHTML",
        "unsanitized",
        "script injection",
        "html injection",
        "user input rendered",
        "escape html",
        "DOM injection",
    ],
    "insecure_storage": [
        "localStorage",
        "local storage",
        "session storage",
        "client-side storage",
        "XSS accessible",
        "token storage",
        "insecure storage",
        "browser storage",
    ],
    "open_redirect": [
        "open redirect",
        "unvalidated redirect",
        "URL injection",
        "user-controlled URL",
        "redirect vulnerability",
        "location assignment",
        "unsafe navigation",
    ],
    "code_injection": [
        "eval",
        "code injection",
        "arbitrary code",
        "dynamic evaluation",
        "code execution",
        "JSON.parse",
        "unsafe parse",
    ],
    "ssrf": [
        "SSRF",
        "server-side request forgery",
        "internal network",
        "user-provided URL",
        "unvalidated URL",
        "request to user URL",
        "outbound request",
    ],
    "credential_leak": [
        "credential leak",
        "log sensitive",
        "logging secret",
        "auth header logged",
        "password in log",
        "token in log",
        "secrets in logs",
    ],
    "unsafe_yaml": [
        "yaml.load",
        "safe_load",
        "unsafe deserialization",
        "arbitrary code",
        "yaml vulnerability",
        "untrusted YAML",
        "deserialize yaml",
    ],
    "ssl_disabled": [
        "verify=False",
        "SSL disabled",
        "certificate verification",
        "insecure connection",
        "TLS disabled",
        "skip verification",
        "MITM",
    ],
    "client_trust": [
        "client-side",
        "hidden field",
        "trust client",
        "manipulate",
        "user-controlled",
        "form manipulation",
        "client validation only",
        "tampering",
    ],
    "async_bug": [
        "await",
        "async",
        "Promise",
        "not awaited",
        "missing await",
        "unresolved promise",
        "promise as value",
        "floating promise",
    ],
    "order_of_operations": [
        "order of operations",
        "discount before tax",
        "tax before discount",
        "calculation order",
        "wrong order",
        "apply first",
        "taxable base",
    ],
    "n_plus_one": [
        "N+1",
        "n plus one",
        "loop query",
        "batch",
        "bulk fetch",
        "individual request",
        "one by one",
        "sequential request",
        "waterfall fetch",
    ],
    "encoding": [
        "encoding",
        "UTF-8",
        "charset",
        "decode error",
        "UnicodeDecodeError",
        "non-ASCII",
        "character encoding",
        "read_csv encoding",
    ],
    "dedup_logic": [
        "deduplication",
        "dedupe",
        "duplicate",
        "unique key",
        "merge logic",
        "data loss",
        "overwrite",
        "subset dedupe",
    ],
    "memory_efficiency": [
        "memory",
        "load all",
        "read entire",
        "memory efficient",
        "streaming",
        "chunk",
        "lazy load",
        "out of memory",
        "full table scan",
    ],
    "date_format": [
        "date format",
        "MM/DD",
        "DD/MM",
        "ambiguous date",
        "date parsing",
        "strftime",
        "strptime",
        "locale",
        "day first",
    ],
    "infinite_loop": [
        "infinite loop",
        "no max retry",
        "while True",
        "unbounded",
        "retry limit",
        "backoff",
        "circuit breaker",
        "unbounded retry",
    ],
}

SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 1.5,
    "major": 1.0,
    "minor": 0.5,
    "suggestion": 0.25,
}


def _severity_weight(severity: str) -> float:
    return SEVERITY_WEIGHTS.get(severity.lower(), 1.0)


def _keyword_match(
    comment_text: str,
    keywords: List[str],
    synonym_group: Optional[str],
) -> bool:
    """True if any keyword matches, or any phrase from the issue's synonym group matches."""
    text_lower = comment_text.lower()
    if any(kw.lower() in text_lower for kw in keywords):
        return True
    if synonym_group and synonym_group in SYNONYM_GROUPS:
        return any(p.lower() in text_lower for p in SYNONYM_GROUPS[synonym_group])
    return False


def _fuzzy_line_match(
    agent_line: int, expected_line: int
) -> Literal["full", "partial", "none"]:
    """Line proximity: within LINE_TOLERANCE -> full; within 8 lines -> partial; else none."""
    dist = abs(agent_line - expected_line)
    if dist <= LINE_TOLERANCE:
        return "full"
    if dist <= 8:
        return "partial"
    return "none"


def _match_score_for_line_tier(
    line_tier: Literal["full", "partial", "none"],
    kw_ok: bool,
    category_match: bool,
    severity_match: bool,
) -> Optional[float]:
    """Return match score in [0, 1] or None if this comment cannot match this issue."""
    if not kw_ok:
        return None

    if line_tier == "none":
        match_score = 0.3
        if category_match:
            match_score += 0.1
        return min(1.0, match_score)

    # full or partial line credit (keyword match)
    match_score = 0.6
    if category_match:
        match_score += 0.2
    if severity_match:
        match_score += 0.2
    match_score = min(1.0, match_score)
    if line_tier == "partial":
        match_score = max(0.0, match_score - 0.1)
    return match_score


def unit_interval_score(x: float) -> float:
    """Clamp a scalar to the closed interval ``[0.0, 1.0]`` (4 decimal places)."""
    return max(0.0, min(1.0, round(float(x), 4)))


# OpenEnv task-score validators require values strictly inside (0, 1), not endpoints.
_STRICT_TASK_SCORE_EPS = 1e-4


def strict_open_unit_score(x: float) -> float:
    """Map ``x`` into ``(0.0, 1.0)`` for wire-format task scores.

    Clamps to the closed unit interval at 4 dp, then affinely maps
    ``[0, 1] -> [eps, 1 - eps]`` so results never equal 0.0 or 1.0.
    """
    eps = _STRICT_TASK_SCORE_EPS
    u = max(0.0, min(1.0, round(float(x), 4)))
    return round(eps + (1.0 - 2.0 * eps) * u, 4)


def grade_comments(
    comments: List[ReviewComment],
    expected_issues: List[dict[str, Any]],
) -> Dict[str, Any]:
    """Grade review comments against expected issues using greedy best-match.

    For each comment, scores against all unmatched issues and keeps the highest
    score. Each expected issue can match at most one comment.

    Returns:
        score: float in ``(0.0, 1.0)`` (strict; OpenEnv task validators)
        sub_score_1: weighted match mass / max possible (recall-style signal)
        sub_score_2: complement of false-positive pressure in ``(0, 1)``
        found_issues: number of distinct expected issues matched
        total_issues: len(expected_issues)
        false_positives: comments that matched no issue
        detail: per-comment rows with comment_line, matched_issue (description or
            None), and points_earned (weighted contribution toward the total)
    """
    if not expected_issues:
        trivial = strict_open_unit_score(1.0)
        return {
            "score": trivial,
            "sub_score_1": trivial,
            "sub_score_2": trivial,
            "found_issues": 0,
            "total_issues": 0,
            "false_positives": 0,
            "detail": [],
        }

    max_points = sum(_severity_weight(issue["severity"]) for issue in expected_issues)

    matched: set[int] = set()
    false_positives = 0
    detail: List[Dict[str, Any]] = []

    for comment in comments:
        best_idx: Optional[int] = None
        best_weighted_score = 0.0

        for idx, issue in enumerate(expected_issues):
            if idx in matched:
                continue

            line_tier = _fuzzy_line_match(comment.line_number, issue["line_number"])
            kw_ok = _keyword_match(
                comment.comment,
                issue["keywords"],
                issue.get("synonym_group"),
            )
            category_match = comment.category == issue["category"]
            severity_match = comment.severity == issue["severity"]

            raw_match = _match_score_for_line_tier(
                line_tier, kw_ok, category_match, severity_match
            )
            if raw_match is None:
                continue

            weight = _severity_weight(issue["severity"])
            weighted = raw_match * weight

            if weighted > best_weighted_score:
                best_weighted_score = weighted
                best_idx = idx

        if best_idx is not None:
            matched.add(best_idx)
            detail.append(
                {
                    "comment_line": comment.line_number,
                    "matched_issue": expected_issues[best_idx]["description"],
                    "points_earned": best_weighted_score,
                }
            )
        else:
            false_positives += 1
            detail.append(
                {
                    "comment_line": comment.line_number,
                    "matched_issue": None,
                    "points_earned": 0.0,
                }
            )

    total_issues = len(expected_issues)
    points_earned = sum(entry["points_earned"] for entry in detail)

    if false_positives <= 0:
        penalty = 0.0
    elif false_positives == 1:
        penalty = 0.05
    else:
        penalty = 0.05 + 0.1 * (false_positives - 1)

    raw_score = (points_earned - penalty) / max_points

    if len(matched) == total_issues and total_issues > 0:
        raw_score += 0.15

    score = strict_open_unit_score(raw_score)
    sub_score_1 = strict_open_unit_score(points_earned / max_points)
    # Penalty grows with false positives; map to a 0–1 “cleanliness” axis.
    sub_score_2 = strict_open_unit_score(
        1.0 - min(1.0, penalty / max(max_points, 1e-6))
    )

    return {
        "score": score,
        "sub_score_1": sub_score_1,
        "sub_score_2": sub_score_2,
        "found_issues": len(matched),
        "total_issues": total_issues,
        "false_positives": false_positives,
        "detail": detail,
    }

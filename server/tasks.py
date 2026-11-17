"""Task definitions with code scenarios and grading criteria.

Each scenario has:
  - diff: the code to review (line numbers in expected_issues refer to this text)
  - file_name, pr_title, pr_description
  - expected_issues: list of issues the agent should find
    Each issue has: line_number, category, severity, keywords (for matching),
    synonym_group (key into server.grader.SYNONYM_GROUPS for flexible phrasing)
"""
import random
from typing import Any, Dict, List, Optional

# --- TASK 1: find-bugs (Easy) ---

BUG_SCENARIOS = [
    {
        "id": "bug-offbyone",
        "file_name": "utils.py",
        "language": "python",
        "pr_title": "Add list helpers for reporting summaries",
        "pr_description": (
            "Adds `get_last_items` for tail slices and `avg` for simple means. "
            "Needed for the weekly metrics email."
        ),
        "diff": '''def get_last_items(items, n):
    """Return the last n items from a list (for dashboard previews)."""
    result = []
    for i in range(len(items) - n, len(items) + 1):
        result.append(items[i])
    return result

def format_name(first, last):
    return (first + " " + last).strip()

def avg(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)
''',
        "expected_issues": [
            {
                "line_number": 4,
                "category": "bug",
                "severity": "critical",
                "keywords": ["off-by-one", "range", "IndexError", "upper bound", "len(items)"],
                "synonym_group": "off_by_one",
                "description": "range() uses len(items)+1 as the stop index, causing IndexError on the last iteration",
            },
            {
                "line_number": 15,
                "category": "bug",
                "severity": "major",
                "keywords": ["division", "zero", "empty", "ZeroDivisionError", "len(numbers)"],
                "synonym_group": "division_zero",
                "description": "avg() divides by len(numbers) with no guard when the list is empty",
            },
        ],
    },
    {
        "id": "bug-wrongvar",
        "file_name": "billing.py",
        "language": "python",
        "pr_title": "Billing totals and batch invoice processing",
        "pr_description": (
            "Introduces `calculate_total` with discount support, user lookup for invoices, "
            "and `process_batch` to run transforms before posting."
        ),
        "diff": '''def calculate_total(prices, tax_rate, discount=0):
    """Compute billed total after discount and tax."""
    subtotal = sum(prices)
    discounted = subtotal * (1 - discount)
    total = discounted + discounted * tax
    return round(total, 2)

def find_user(users, user_id):
    for user in users:
        if user["id"] == user_id:
            return user

def transform(item):
    return item * 2
def process_batch(items):
    results = []
    for item in items:
        result = transform(item)
        results.append(results)
    return results
''',
        "expected_issues": [
            {
                "line_number": 5,
                "category": "bug",
                "severity": "critical",
                "keywords": ["tax", "tax_rate", "NameError", "undefined", "variable"],
                "synonym_group": "wrong_variable",
                "description": "Uses undefined name `tax` instead of parameter `tax_rate`",
            },
            {
                "line_number": 9,
                "category": "bug",
                "severity": "minor",
                "keywords": ["None", "not found", "return", "implicit", "missing"],
                "synonym_group": "missing_return",
                "description": "find_user falls through without return or raise when no user matches",
            },
            {
                "line_number": 19,
                "category": "bug",
                "severity": "critical",
                "keywords": ["append", "results", "result", "list", "wrong"],
                "synonym_group": "append_wrong",
                "description": "Appends the `results` list instead of the per-item `result` value",
            },
        ],
    },
    {
        "id": "bug-js-array",
        "file_name": "helpers.js",
        "language": "javascript",
        "pr_title": "Array utility helpers",
        "pr_description": (
            "Adds shared helpers for dashboard CSV previews: dedupe rows, simple averages, "
            "and one-level flattening before charting."
        ),
        "diff": '''/**
 * Shared array utilities for reporting dashboards and CSV previews.
 */

function removeDuplicates(arr) {
  return arr.filter(function (item, idx) {
    return arr.indexOf(item) == idx;
  });
}

function getAverage(nums) {
  var sum = 0;
  for (var i = 0; i < nums.length; i++) {
    sum += nums[i];
  }
  return sum / nums.length;
}

function flattenOnce(arr) {
  var result = [];
  for (var i = 0; i < arr.length; i++) {
    var item = arr[i];
    if (Array.isArray(item)) {
      arr.concat(item);
    } else {
      result.push(item);
    }
  }
  return result;
}
''',
        "expected_issues": [
            {
                "line_number": 7,
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "indexOf",
                    "filter",
                    "equality",
                    "strict",
                    "coercion",
                    "duplicate",
                    "==",
                ],
                "synonym_group": "type_coercion",
                "description": "Uses loose `==` when comparing `indexOf` to the index, inviting type coercion bugs",
            },
            {
                "line_number": 16,
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "average",
                    "empty",
                    "NaN",
                    "divide",
                    "length",
                    "guard",
                    "zero",
                ],
                "synonym_group": "division_zero",
                "description": "Divides by `nums.length` with no empty-array guard, producing NaN for empty input",
            },
            {
                "line_number": 24,
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "concat",
                    "flatten",
                    "reassign",
                    "result",
                    "ignored",
                    "mutation",
                    "accumulator",
                ],
                "synonym_group": "append_wrong",
                "description": "Calls `arr.concat(item)` without assigning to `result`, so nested arrays never merge",
            },
        ],
    },
    {
        "id": "bug-dict-mutation",
        "file_name": "cache.py",
        "language": "python",
        "pr_title": "Simple in-memory cache with TTL",
        "pr_description": (
            "Introduces a lightweight TTL cache for worker memoization: set/get with optional TTL, "
            "plus a periodic cleanup helper for ops cron."
        ),
        "diff": '''"""Simple in-memory cache with TTL for worker-side memoization."""

from time import time


class TTLCache:
    """Minimal dict-backed cache with per-key expiry."""

    def __init__(self, default_ttl_sec: float = 300.0, store={}):
        self.default_ttl_sec = default_ttl_sec
        self.store = store

    def set(self, key, value, ttl_sec: float | None = None) -> None:
        ttl = self.default_ttl_sec if ttl_sec is None else ttl_sec
        self.store[key] = {"value": value, "expire_at": time() + ttl}

    def get(self, key):
        entry = self.store.get(key)
        if entry is None:
            return None
        if entry["expire_at"] <= time():
            return None
        return entry["value"]

    def cleanup(self) -> None:
        now = time()
        for key in self.store:
            if self.store[key]["expire_at"] <= now:
                del self.store[key]
''',
        "expected_issues": [
            {
                "line_number": 9,
                "category": "bug",
                "severity": "critical",
                "keywords": [
                    "default",
                    "mutable",
                    "dict",
                    "__init__",
                    "shared",
                    "store",
                    "argument",
                ],
                "synonym_group": "mutable_default",
                "description": "Mutable default `store={}` shares backing dict across instances",
            },
            {
                "line_number": 22,
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "expired",
                    "TTL",
                    "delete",
                    "stale",
                    "get",
                    "None",
                    "cleanup",
                ],
                "synonym_group": "stale_data",
                "description": "Returns None for expired entries but never removes the key, so dead keys accumulate",
            },
            {
                "line_number": 29,
                "category": "bug",
                "severity": "critical",
                "keywords": [
                    "iterate",
                    "delete",
                    "dict",
                    "RuntimeError",
                    "mutation",
                    "loop",
                    "cleanup",
                ],
                "synonym_group": "dict_mutation",
                "description": "Deletes keys from `self.store` while iterating it, which raises RuntimeError in Python 3",
            },
        ],
    },
]

# --- TASK 2: security-review (Medium) ---

SECURITY_SCENARIOS = [
    {
        "id": "sec-sqli",
        "file_name": "db.py",
        "language": "python",
        "pr_title": "User lookup and token persistence helpers",
        "pr_description": (
            "SQLite helpers for login flows: fetch user by username, legacy password hash, "
            "and store opaque session tokens."
        ),
        "diff": '''import sqlite3
import hashlib

DB_PATH = "app.db"
API_KEY = "sk-prod-a8f3bc91d4e567f0123456789abcdef0"


def get_connection():
    return sqlite3.connect(DB_PATH)


def get_user(username):
    conn = get_connection()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    row = conn.execute(query).fetchone()
    conn.close()
    return row


def hash_password(password):
    return hashlib.md5(password.encode("utf-8")).hexdigest()


def save_token(user_id, token):
    conn = get_connection()
    conn.execute(
        f"INSERT INTO tokens (user_id, token) VALUES ({user_id}, '{token}')"
    )
    conn.commit()
    conn.close()
''',
        "expected_issues": [
            {
                "line_number": 5,
                "category": "security",
                "severity": "critical",
                "keywords": ["API_KEY", "hardcoded", "secret", "credential", "source"],
                "synonym_group": "hardcoded_secret",
                "description": "Production API key committed as a string literal",
            },
            {
                "line_number": 14,
                "category": "security",
                "severity": "critical",
                "keywords": ["SQL injection", "f-string", "parameterized", "username", "query"],
                "synonym_group": "sql_injection",
                "description": "User-controlled username interpolated into SQL via f-string",
            },
            {
                "line_number": 21,
                "category": "security",
                "severity": "major",
                "keywords": ["MD5", "password", "hash", "weak", "bcrypt"],
                "synonym_group": "weak_hash",
                "description": "Passwords hashed with MD5, which is unsuitable for credential storage",
            },
            {
                "line_number": 27,
                "category": "security",
                "severity": "critical",
                "keywords": ["SQL injection", "INSERT", "token", "parameterized", "f-string"],
                "synonym_group": "sql_injection",
                "description": "Token and user_id concatenated into SQL, enabling injection",
            },
        ],
    },
    {
        "id": "sec-pathtraversal",
        "file_name": "fileserver.py",
        "language": "python",
        "pr_title": "User file upload and management routes",
        "pr_description": (
            "Flask routes so customers can upload documents, download them later, "
            "or remove outdated copies from the shared upload directory."
        ),
        "diff": '''import os
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
UPLOAD_DIR = "/var/uploads"
SECRET_KEY = "mysecretkey123"

app.config["SECRET_KEY"] = SECRET_KEY


@app.route("/download")
def download():
    filepath = os.path.join(UPLOAD_DIR, request.args.get("file"))
    return send_file(filepath)


@app.route("/delete")
def delete():
    filepath = UPLOAD_DIR + "/" + request.args.get("file")
    os.remove(filepath)
    return jsonify({"status": "deleted"})


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files["file"]
    dest = UPLOAD_DIR
    f.save(os.path.join(dest, f.filename))
    return jsonify({"status": "uploaded"})
''',
        "expected_issues": [
            {
                "line_number": 6,
                "category": "security",
                "severity": "critical",
                "keywords": ["SECRET_KEY", "hardcoded", "Flask", "secret", "config"],
                "synonym_group": "hardcoded_secret",
                "description": "Flask secret key is a fixed string in the repository",
            },
            {
                "line_number": 13,
                "category": "security",
                "severity": "critical",
                "keywords": ["path traversal", "filename", "join", "sanitize", "../"],
                "synonym_group": "path_traversal",
                "description": "Download path built from unsanitized query param, allowing traversal",
            },
            {
                "line_number": 19,
                "category": "security",
                "severity": "critical",
                "keywords": ["path traversal", "delete", "remove", "sanitize", "../"],
                "synonym_group": "path_traversal",
                "description": "Delete builds path from unsanitized query input, enabling traversal",
            },
            {
                "line_number": 28,
                "category": "security",
                "severity": "major",
                "keywords": ["upload", "filename", "secure_filename", "path", "traversal"],
                "synonym_group": "path_traversal",
                "description": "Uploaded files saved under raw client-provided filename",
            },
        ],
    },
    {
        "id": "sec-js-xss",
        "file_name": "api.js",
        "language": "javascript",
        "pr_title": "User profile rendering API",
        "pr_description": (
            "Client-side helpers for the legacy admin console: bio rendering, session token "
            "persistence, profile navigation, and preference parsing."
        ),
        "diff": '''/**
 * User profile rendering API for the legacy admin console.
 */

function renderProfileBio(container, bioHtml) {
  container.innerHTML = '<div class="profile-bio">' + bioHtml + "</div>";
}

function saveAuthToken(jwt) {
  window.localStorage.setItem("session_jwt", jwt);
}

function navigateToProfile(userId) {
  window.location = "/profile/" + userId;
}

function parsePreferences(raw) {
  return eval("(" + raw + ")");
}

export { navigateToProfile, parsePreferences, renderProfileBio, saveAuthToken };
''',
        "expected_issues": [
            {
                "line_number": 6,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "innerHTML",
                    "XSS",
                    "bio",
                    "HTML",
                    "injection",
                    "sanitize",
                    "escape",
                ],
                "synonym_group": "xss",
                "description": "Assigns unsanitized user bio HTML to `innerHTML`, enabling cross-site scripting",
            },
            {
                "line_number": 10,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "localStorage",
                    "JWT",
                    "token",
                    "session",
                    "XSS",
                    "client",
                    "storage",
                ],
                "synonym_group": "insecure_storage",
                "description": "Persists JWT in `localStorage`, readable by any XSS on the origin",
            },
            {
                "line_number": 14,
                "category": "security",
                "severity": "major",
                "keywords": [
                    "redirect",
                    "location",
                    "URL",
                    "userId",
                    "open redirect",
                    "navigation",
                    "path",
                ],
                "synonym_group": "open_redirect",
                "description": "Builds navigation URL from raw `userId` without validation or encoding",
            },
            {
                "line_number": 18,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "eval",
                    "JSON",
                    "parse",
                    "preferences",
                    "code execution",
                    "injection",
                    "dynamic",
                ],
                "synonym_group": "code_injection",
                "description": "Uses `eval` to parse preference JSON instead of `JSON.parse`, enabling code injection",
            },
        ],
    },
    {
        "id": "sec-ssrf",
        "file_name": "webhook_handler.py",
        "language": "python",
        "pr_title": "Webhook notification handler",
        "pr_description": (
            "Outbound relay for partner webhooks: POST JSON to callback URLs from the request, "
            "audit logging, optional YAML routing config, and permissive TLS for legacy endpoints."
        ),
        "diff": '''"""Webhook notification handler for outbound partner callbacks."""

import logging
from typing import Any

import requests
import yaml

LOGGER = logging.getLogger(__name__)
AUDIT_LOG = "/var/log/webhook_dispatch.log"


def forward_event(target_url: str, body: dict[str, Any], auth: str) -> requests.Response:
    """POST JSON payload to partner-supplied callback URL."""
    response = requests.post(
        target_url,
        json=body,
        headers={"Authorization": auth, "Content-Type": "application/json"},
        timeout=30,
        verify=False,
    )
    with open(AUDIT_LOG, "a", encoding="utf-8") as fh:
        fh.write(
            f"url={target_url} status={response.status_code} auth={auth} body={body}\\n"
        )
    return response


def load_partner_config(yaml_blob: str) -> dict[str, Any]:
    """Parse YAML partner routing table (admin-editable)."""
    return yaml.load(yaml_blob, Loader=yaml.Loader)


def notify_from_payload(incoming: dict[str, Any], service_token: str) -> requests.Response:
    """Read URL from posted JSON and relay the event."""
    callback = incoming["callback_url"]
    return forward_event(callback, incoming.get("payload", {}), service_token)
''',
        "expected_issues": [
            {
                "line_number": 16,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "SSRF",
                    "requests.post",
                    "URL",
                    "user",
                    "callback",
                    "forgery",
                    "internal",
                ],
                "synonym_group": "ssrf",
                "description": "Issues HTTP requests to caller-controlled URLs with no allowlist or SSRF protections",
            },
            {
                "line_number": 24,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "log",
                    "Authorization",
                    "secret",
                    "token",
                    "audit",
                    "credential",
                    "sensitive",
                ],
                "synonym_group": "credential_leak",
                "description": "Writes full `Authorization` header and body into a flat text audit log",
            },
            {
                "line_number": 31,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "yaml.load",
                    "safe_load",
                    "deserialize",
                    "YAML",
                    "untrusted",
                    "Loader",
                ],
                "synonym_group": "unsafe_yaml",
                "description": "Uses `yaml.load` with unsafe `Loader` on user-editable YAML instead of `safe_load`",
            },
            {
                "line_number": 20,
                "category": "security",
                "severity": "major",
                "keywords": [
                    "verify=False",
                    "TLS",
                    "SSL",
                    "certificate",
                    "MITM",
                    "requests",
                    "insecure",
                ],
                "synonym_group": "ssl_disabled",
                "description": "Disables TLS certificate verification on outbound `requests.post` calls",
            },
        ],
    },
]

# --- TASK 3: comprehensive-review (Hard) ---

COMPREHENSIVE_SCENARIOS = [
    {
        "id": "comp-orderservice",
        "file_name": "order_service.py",
        "language": "python",
        "pr_title": "Order pipeline: validation, discounts, payment, summaries",
        "pr_description": (
            "End-to-end order handling including DB-backed totals, promo codes, synchronous "
            "payment calls, background notifications, and per-user revenue rollup for ops."
        ),
        "diff": '''import json
import time
import requests
from threading import Thread

DB_URL = "postgresql://admin:admin123@db:5432/orders"
PAYMENT_KEY = "pk_live_abc123secret"


def create_order(user_id, items):
    order = {
        "user_id": user_id,
        "items": items,
        "total": 0,
        "status": "pending",
    }
    for item in items:
        order["total"] += item["price"] * item["qty"]
    if order["total"] > 0:
        order["status"] = "validated"
    return order


def apply_discount(order, code):
    discounts = {"SAVE10": 0.10, "SAVE20": 0.20, "FREE": 1.0}
    if code in discounts:
        pct = discounts[code]
        order["total"] = order["total"] * (1 - pct)
    return order


def process_payment(order):
    resp = requests.post(
        "https://pay.example.com/charge",
        json={"amount": order["total"], "key": PAYMENT_KEY},
        timeout=30,
    )
    if resp.status_code == 200:
        order["status"] = "paid"
    return order


def send_notification(user_id, message):
    def _send():
        time.sleep(0.1)
        requests.post(
            "https://notify.example.com/send",
            json={"user": user_id, "msg": message},
        )

    Thread(target=_send).start()


def process_orders(order_list):
    results = []
    for order in order_list:
        try:
            paid = process_payment(order)
            results.append(paid)
        except:
            pass
    return results


def get_order_summary(orders):
    summary = {}
    for o in orders:
        uid = o["user_id"]
        if uid not in summary:
            summary[uid] = {"count": 0, "total": 0}
        summary[uid]["count"] += 1
        summary[uid]["total"] += o["total"]
    sorted_users = sorted(summary.items(), key=lambda x: x[1]["total"])
    return sorted_users[:10]
''',
        "expected_issues": [
            {
                "line_number": 6,
                "category": "security",
                "severity": "critical",
                "keywords": ["DB_URL", "password", "hardcoded", "credential", "postgresql"],
                "synonym_group": "hardcoded_secret",
                "description": "Database connection string with embedded username and password",
            },
            {
                "line_number": 7,
                "category": "security",
                "severity": "critical",
                "keywords": ["PAYMENT_KEY", "hardcoded", "secret", "API", "live"],
                "synonym_group": "hardcoded_secret",
                "description": "Live payment API key stored as a module-level constant",
            },
            {
                "line_number": 18,
                "category": "logic",
                "severity": "major",
                "keywords": ["validation", "negative", "price", "quantity", "qty"],
                "synonym_group": "wrong_variable",
                "description": "No checks that price or qty are non-negative before totaling",
            },
            {
                "line_number": 25,
                "category": "logic",
                "severity": "major",
                "keywords": ["FREE", "discount", "100%", "zero", "promo"],
                "synonym_group": "wrong_variable",
                "description": "Promo map includes FREE at 100% off with no safeguards",
            },
            {
                "line_number": 46,
                "category": "bug",
                "severity": "minor",
                "keywords": ["notification", "thread", "errors", "silent", "response"],
                "synonym_group": "bare_except",
                "description": "Background notification thread ignores HTTP failures and exceptions",
            },
            {
                "line_number": 60,
                "category": "bug",
                "severity": "critical",
                "keywords": ["bare except", "pass", "payment", "swallow", "errors"],
                "synonym_group": "bare_except",
                "description": "Bare except with pass hides all payment processing failures",
            },
            {
                "line_number": 73,
                "category": "performance",
                "severity": "minor",
                "keywords": ["sorted", "heapq", "nlargest", "top", "efficiency"],
                "synonym_group": "wrong_variable",
                "description": "Full sort of all users to return only ten largest totals",
            },
        ],
    },
    {
        "id": "comp-inventoryapi",
        "file_name": "inventory_api.py",
        "language": "python",
        "pr_title": "Internal inventory sync API",
        "pr_description": (
            "FastAPI-style helpers for warehouse sync: JWT minting, bulk stock updates, "
            "and a cache helper for hot SKUs. Intended for trusted internal callers only."
        ),
        "diff": '''import json
import pickle
import subprocess
import jwt
from datetime import datetime, timedelta

JWT_SECRET = "warehouse-jwt-secret-do-not-share"
ADMIN_TOKEN = "Bearer static-admin-token-001"


def mint_service_token(service_id: str) -> str:
    payload = {
        "sub": service_id,
        "exp": datetime.utcnow() + timedelta(hours=12),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def apply_stock_delta(sku: str, delta_json: str):
    delta = json.loads(delta_json)
    new_qty = int(delta["qty"])
    cmd = f"inventory-cli adjust --sku {sku} --qty {new_qty}"
    subprocess.run(cmd, shell=True, check=True)
    return {"sku": sku, "qty": new_qty}


def load_user_preferences(blob: bytes):
    return pickle.loads(blob)


def get_hot_skus(access_count: dict):
    pairs = sorted(access_count.items(), key=lambda kv: kv[1], reverse=True)
    top = pairs[:50]
    return [sku for sku, _ in top]


def merge_shipments(left: dict, right: dict):
    for k, v in right.items():
        if k in left and isinstance(left[k], dict) and isinstance(v, dict):
            merge_shipments(left[k], v)
        else:
            left[k] = v
    return left
''',
        "expected_issues": [
            {
                "line_number": 7,
                "category": "security",
                "severity": "critical",
                "keywords": ["JWT_SECRET", "hardcoded", "secret", "signing", "token"],
                "synonym_group": "hardcoded_secret",
                "description": "JWT signing secret is a hardcoded string in source",
            },
            {
                "line_number": 8,
                "category": "security",
                "severity": "critical",
                "keywords": ["ADMIN_TOKEN", "Bearer", "static", "credential", "hardcoded"],
                "synonym_group": "hardcoded_secret",
                "description": "Long-lived admin bearer token embedded in code",
            },
            {
                "line_number": 23,
                "category": "security",
                "severity": "critical",
                "keywords": ["subprocess", "shell", "injection", "command", "sku"],
                "synonym_group": "sql_injection",
                "description": "Shell=True with interpolated sku/qty allows command injection",
            },
            {
                "line_number": 28,
                "category": "security",
                "severity": "critical",
                "keywords": ["pickle", "deserialize", "RCE", "untrusted", "loads"],
                "synonym_group": "sql_injection",
                "description": "pickle.loads on arbitrary bytes enables remote code execution",
            },
            {
                "line_number": 32,
                "category": "performance",
                "severity": "minor",
                "keywords": ["sorted", "heapq", "nlargest", "hot", "skus"],
                "synonym_group": "wrong_variable",
                "description": "Sorts every SKU by access count instead of a bounded heap for top 50",
            },
            {
                "line_number": 40,
                "category": "bug",
                "severity": "major",
                "keywords": ["recursion", "cycle", "dict", "merge", "stack"],
                "synonym_group": "append_wrong",
                "description": "Recursive dict merge can infinite-loop or blow the stack on cyclic structures",
            },
        ],
    },
    {
        "id": "comp-js-ecommerce",
        "file_name": "checkout.js",
        "language": "javascript",
        "pr_title": "Checkout flow handler",
        "pr_description": (
            "Implements checkout totals with regional tax, client-submitted discount percentage, "
            "line-item hydration, inventory search filter building, and order submission."
        ),
        "diff": '''/**
 * Checkout flow handler for the storefront checkout wizard.
 * Totals, tax, discounts, line items, and submission.
 */

function readClientDiscountPercent() {
  const el = document.querySelector('input[type="hidden"][name="discount_pct"]');
  return Number.parseFloat(el.value, 10);
}

async function fetchTaxRate(regionCode) {
  const res = await fetch(`/api/v1/tax?region=${encodeURIComponent(regionCode)}`);
  if (!res.ok) throw new Error("tax service error");
  const data = await res.json();
  return data.rate;
}

async function calculateCheckoutTotal(subtotal, regionCode) {
  const discountPct = readClientDiscountPercent();
  const taxFetch = fetchTaxRate(regionCode);
  const taxRate = taxFetch;
  const withTax = subtotal * (1 + taxRate);
  const afterDiscount = withTax * (1 - discountPct / 100);
  return Math.round(afterDiscount * 100) / 100;
}

async function hydrateProducts(productIds) {
  const rows = [];
  for (const id of productIds) {
    const r = await fetch(`/api/v1/products/${encodeURIComponent(id)}`);
    rows.push(await r.json());
  }
  return rows;
}

function buildSkuSearchFilter(userQuery) {
  return `status = 'shipped' AND (sku LIKE '%${userQuery}%' OR name LIKE '%${userQuery}%')`;
}

export async function submitOrder(orderPayload) {
  try {
    const res = await fetch("/api/v1/orders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(orderPayload),
    });
    return await res.json();
  } catch (_err) {
    return { ok: false, message: "Something went wrong" };
  }
}

export { buildSkuSearchFilter, calculateCheckoutTotal, hydrateProducts, readClientDiscountPercent };
''',
        "expected_issues": [
            {
                "line_number": 7,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "hidden",
                    "discount",
                    "client",
                    "trust",
                    "tamper",
                    "form",
                    "server-side",
                ],
                "synonym_group": "client_trust",
                "description": "Reads discount percentage from a hidden form field, so shoppers can manipulate totals",
            },
            {
                "line_number": 21,
                "category": "bug",
                "severity": "critical",
                "keywords": [
                    "await",
                    "Promise",
                    "async",
                    "tax",
                    "fetchTaxRate",
                    "numeric",
                    "NaN",
                ],
                "synonym_group": "async_bug",
                "description": "Assigns the pending Promise from `fetchTaxRate` to `taxRate` without `await`",
            },
            {
                "line_number": 22,
                "category": "logic",
                "severity": "major",
                "keywords": [
                    "discount",
                    "tax",
                    "order",
                    "subtotal",
                    "taxable",
                    "calculation",
                    "sequence",
                ],
                "synonym_group": "order_of_operations",
                "description": "Applies tax to the full subtotal then discount, charging tax on the undiscounted base",
            },
            {
                "line_number": 30,
                "category": "performance",
                "severity": "major",
                "keywords": [
                    "loop",
                    "fetch",
                    "sequential",
                    "batch",
                    "N+1",
                    "products",
                    "network",
                ],
                "synonym_group": "n_plus_one",
                "description": "Fetches each product in a serial loop instead of batching or parallel bulk requests",
            },
            {
                "line_number": 37,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "SQL",
                    "injection",
                    "LIKE",
                    "string",
                    "concat",
                    "userQuery",
                    "filter",
                ],
                "synonym_group": "sql_injection",
                "description": "Interpolates raw search text into a SQL-like filter string for downstream APIs",
            },
            {
                "line_number": 48,
                "category": "style",
                "severity": "minor",
                "keywords": [
                    "catch",
                    "generic",
                    "error",
                    "logging",
                    "swallow",
                    "Something went wrong",
                    "opaque",
                ],
                "synonym_group": "bare_except",
                "description": "Catches all failures and returns a generic message with no logging or error detail",
            },
        ],
    },
    {
        "id": "comp-data-pipeline",
        "file_name": "etl_pipeline.py",
        "language": "python",
        "pr_title": "Daily ETL data pipeline",
        "pr_description": (
            "Nightly job: ingest vendor CSV, dedupe customers, filter paid orders, load to "
            "warehouse via psycopg2, normalize timestamps, and retry steps until success."
        ),
        "diff": '''"""Daily ETL data pipeline from the landing zone into the sales warehouse."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import psycopg2


def read_orders_snapshot(csv_path: str) -> pd.DataFrame:
    """Load the daily export; assumes UTF-8 encoded vendor files."""
    return pd.read_csv(csv_path)


def dedupe_by_email(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per email for downstream warehouse keys."""
    return df.drop_duplicates(subset=["email"], keep="last")


def load_paid_orders_only(export_path: str) -> pd.DataFrame:
    """Filter to paid rows — reads full export into memory first for simplicity."""
    entire = pd.read_csv(export_path)
    paid = entire[entire["payment_status"] == "paid"]
    return paid


def sales_database_url() -> str:
    """Connection string used by nightly batch (rotated quarterly in ops runbooks)."""
    return "postgresql://etl_batch:EtLB@tch_S3cr3t!@warehouse.prod.internal:5432/sales_dw"


def normalize_order_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_ts"] = pd.to_datetime(out["order_ts"], format="%m/%d/%Y %H:%M:%S")
    return out


def run_until_success(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    while True:
        try:
            return operation(*args, **kwargs)
        except Exception:
            time.sleep(2.0)
''',
        "expected_issues": [
            {
                "line_number": 15,
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "read_csv",
                    "encoding",
                    "UTF-8",
                    "Unicode",
                    "decode",
                    "vendor",
                    "crash",
                ],
                "synonym_group": "encoding",
                "description": "Reads CSV without `encoding` or error handling, so non-UTF-8 bytes abort the pipeline",
            },
            {
                "line_number": 20,
                "category": "logic",
                "severity": "major",
                "keywords": [
                    "dedupe",
                    "email",
                    "duplicate",
                    "merge",
                    "data loss",
                    "unique",
                    "person",
                ],
                "synonym_group": "dedup_logic",
                "description": "Deduplicates only on email, collapsing distinct people or losing newer field values",
            },
            {
                "line_number": 25,
                "category": "performance",
                "severity": "major",
                "keywords": [
                    "memory",
                    "read_csv",
                    "full",
                    "filter",
                    "chunk",
                    "streaming",
                    "pandas",
                ],
                "synonym_group": "memory_efficiency",
                "description": "Loads the entire export into memory before filtering instead of predicate pushdown or chunks",
            },
            {
                "line_number": 32,
                "category": "security",
                "severity": "critical",
                "keywords": [
                    "postgresql",
                    "password",
                    "credential",
                    "connection string",
                    "hardcoded",
                    "secret",
                    "warehouse",
                ],
                "synonym_group": "hardcoded_secret",
                "description": "Embeds warehouse username and password directly in a returned connection string",
            },
            {
                "line_number": 37,
                "category": "bug",
                "severity": "major",
                "keywords": [
                    "datetime",
                    "format",
                    "MM/DD",
                    "DD/MM",
                    "ambiguous",
                    "locale",
                    "parsing",
                ],
                "synonym_group": "date_format",
                "description": "Forces `%m/%d/%Y` parsing though vendor rows may be DD/MM, silently shifting dates",
            },
            {
                "line_number": 42,
                "category": "logic",
                "severity": "critical",
                "keywords": [
                    "while True",
                    "retry",
                    "unbounded",
                    "infinite",
                    "backoff",
                    "circuit breaker",
                    "sleep",
                ],
                "synonym_group": "infinite_loop",
                "description": "Retries forever on `while True` with no cap or jitter, stalling the job on persistent errors",
            },
        ],
    },
]


# Primary rubric task ids (easy / medium / hard)
TASKS: Dict[str, List[Dict[str, Any]]] = {
    "easy": BUG_SCENARIOS,
    "medium": SECURITY_SCENARIOS,
    "hard": COMPREHENSIVE_SCENARIOS,
}

# Legacy names accepted by ``get_scenario`` / ``reset``
TASK_ALIASES: Dict[str, str] = {
    "find-bugs": "easy",
    "security-review": "medium",
    "comprehensive-review": "hard",
}


def resolve_task_name(task_name: str) -> str:
    """Map legacy task names to canonical ``easy`` / ``medium`` / ``hard``."""
    if task_name in TASKS:
        return task_name
    if task_name in TASK_ALIASES:
        return TASK_ALIASES[task_name]
    known = sorted(set(TASKS) | set(TASK_ALIASES))
    raise ValueError(f"Unknown task: {task_name!r}. Available: {known}")


def get_scenario(
    task_name: str,
    seed: Optional[int] = None,
    scenario_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Pick a scenario for the given task (reproducible with ``seed`` / ``scenario_index``)."""
    canonical = resolve_task_name(task_name)
    scenarios = TASKS[canonical]
    rng: random.Random = random.Random(seed) if seed is not None else random
    if scenario_index is not None:
        return scenarios[int(scenario_index) % len(scenarios)]
    return rng.choice(scenarios)


def get_all_task_names() -> List[str]:
    """Canonical task ids for listings and rubrics."""
    return ["easy", "medium", "hard"]

"""BEKE authentication — login, guest rate limiting, session management.

Functions (build on each other bottom-up):
  check_password()  — verify plaintext against bcrypt hash        [COMPLETED]
  load_users()      — read users.yaml → dict
  _guest_count()    — read/increment shared guest query counter
  login_page()      — Streamlit login form + guest button
  require_auth()    — gate for app.py — call at top of main()
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

import bcrypt                    # pip install bcrypt
import yaml                     # pip install pyyaml
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USERS_PATH        = Path("users.yaml")
GUEST_COUNTER     = Path("/tmp/beke_guest_count.json")
GUEST_DAILY_LIMIT = 3


# ---------------------------------------------------------------------------
# check_password  (COMPLETED by user)
# ---------------------------------------------------------------------------

def check_password(username: str, password: str, users: dict) -> bool:
    """Return True if username exists in users and password matches its bcrypt hash."""
    user = users.get(username)
    if user is None:
        return False
    stored_hash = user["password_hash"]
    return bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash.encode("utf-8"),
    )


# ---------------------------------------------------------------------------
# Scaffold 2 of 5 — load_users
# load_users: read user credentials from a YAML file.
# Input:  path — Path to the YAML credentials file (default: users.yaml)
# Output: dict — {username: {name: ..., password_hash: ...}, ...}
# ---------------------------------------------------------------------------

def load_users(path: Path = USERS_PATH) -> dict:
    """Load user credentials from a YAML file. Returns empty dict if file missing."""

    # Step 1: guard — return empty dict if the file doesn't exist
    # Hint: Path exposes a boolean method to check if the path points to a real file
    # Background: returning {} on missing file means logins fail gracefully —
    #             no crash, just "invalid credentials" for everyone
    if not path.exists():                                              # (easy)
        # Answer: path.exists()
        return {}

    # Step 2: open the file and parse YAML into a Python dict
    # Hint: the yaml module has two load functions — one safe, one dangerous
    # Background: yaml.load() can deserialise arbitrary Python objects (code execution);
    #             the safe variant only allows basic types (str, int, list, dict)
    with open(path, "r") as f:
        data = yaml.safe_load(f)                                          # (think)
    # Answer: yaml.safe_load(f)

    # Step 3: guard — YAML file might be empty (returns None)
    return data or {}


# ---------------------------------------------------------------------------
# Scaffold 3 of 5 — _guest_count
# _guest_count: read and optionally increment the shared guest query counter.
# Input:  increment — bool (True to add 1 to today's count)
# Output: int — the current count for today (after any increment)
#
# File format: {"date": "2026-06-07", "count": 2}
# Resets automatically when the date changes.
# ---------------------------------------------------------------------------

def _guest_count(increment: bool = False) -> int:
    """Return today's guest query count. If increment=True, add 1 first."""

    today = date.today().isoformat()

    # Step 1: read existing counter file, or start fresh
    # Hint: the json module mirrors yaml — it has a function to parse a file handle
    # Background: json.load(f) reads from a file object; json.loads(s) reads from a string
    try:
        with open(GUEST_COUNTER, "r") as f:
            data = json.load(f)                                      # (easy)
        # Answer: json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"date": today, "count": 0}

    # Step 2: reset counter if it's a new day
    if data["date"] != today:                                                         # (think)
        data = {"date": today, "count": 0}

    # Step 3: increment if requested, then persist
    # Hint: json has a write counterpart to json.load — same naming pattern
    # Background: json.dump(obj, f) writes to a file object; json.dumps(obj) returns a string
    if increment:
        data["count"] += 1
        with open(GUEST_COUNTER, "w") as f:
            json.dump(data, f)                                       # (easy)
        # Answer: json.dump(data, f)

    return data["count"]


# ---------------------------------------------------------------------------
# Scaffold 4 of 5 — login_page
# login_page: render a Streamlit login form with username/password + guest option.
# Input:  none (reads from Streamlit widgets)
# Output: None (sets st.session_state["role"] and st.session_state["username"])
# ---------------------------------------------------------------------------

def login_page() -> None:
    """Render login form. Sets session_state role to 'user' or 'guest' on success."""

    st.markdown("### Login")

    # Step 1: create a Streamlit form (batches inputs until submit is clicked)
    # Hint: st has a context-manager widget that groups inputs and delays reruns
    # Background: without a form, every keystroke in the password field reruns the
    #             entire app — the form batches until the user clicks submit
    with st.form("login_form"):                                          # (think)
    # Answer: st.form("login_form")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted and username and password:
        users = load_users()
        # Hint: you built a function in scaffold 1 that verifies a password against a hash
        # Background: check_password(username, password, users) returns True/False —
        #             all three args are already defined in this scope
        if check_password(username, password, users):                                                     # (easy)
        # Answer: check_password(username, password, users)
            st.session_state["role"] = "user"
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown("---")

    # Step 2: guest access button with rate limit check
    if st.button("Continue as Guest"):
        count = _guest_count()
        if count >= GUEST_DAILY_LIMIT:                                                     # (think)
        # Answer: count >= GUEST_DAILY_LIMIT
            st.error(
                f"Guest limit reached ({GUEST_DAILY_LIMIT} queries/day). "
                "Please log in for unlimited access."
            )
        else:
            st.session_state["role"] = "guest"
            st.session_state["username"] = "guest"
            st.rerun()


# ---------------------------------------------------------------------------
# Scaffold 5 of 5 — require_auth
# require_auth: gate function — call at top of main() to block unauthenticated access.
# Input:  none (reads st.session_state)
# Output: None (renders login page and calls st.stop() if not authenticated)
# ---------------------------------------------------------------------------

def require_auth() -> None:
    """Block app rendering until user logs in or enters as guest."""

    # Step 1: check if a role is already set in the session
    # Hint: st.session_state behaves like a dict — use .get() for safe access
    # Background: .get("key") returns None (falsy) if key was never set,
    #             avoiding a KeyError that [] would raise on missing keys
    if st.session_state.get("role"):                                                         # (think)
    # Answer: st.session_state.get("role")
        return  # already authenticated — let the app proceed

    # Step 2: not logged in — show the login page and halt
    # Hint: Streamlit has a function that stops script execution mid-run
    # Background: without halting, the rest of main() would render below
    #             the login form — exposing the query box to unauthenticated users
    login_page()
    st.stop()                                                        # (easy)
    # Answer: st.stop()


def is_guest() -> bool:
    """Return True if current session is a guest (used for feature gating)."""
    return st.session_state.get("role") == "guest"


def guest_query_allowed() -> bool:
    """Check if guest has queries remaining. Increments counter if yes."""
    count = _guest_count(increment=True)
    return count <= GUEST_DAILY_LIMIT


# ---------------------------------------------------------------------------
# Smoke test (only tests non-Streamlit functions)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # check_password
    test_hash = bcrypt.hashpw(b"testpass123", bcrypt.gensalt()).decode()
    fake_users = {"alice": {"name": "Alice", "password_hash": test_hash}}
    assert check_password("alice", "testpass123", fake_users) is True
    assert check_password("alice", "wrongpass", fake_users) is False
    assert check_password("nobody", "testpass123", fake_users) is False
    print("check_password: PASS")

    # load_users
    users = load_users()
    print(f"load_users: loaded {len(users)} user(s) — {list(users.keys())}")

    # guest counter
    count = _guest_count()
    print(f"guest_count: {count} queries used today")

    print("\nAll non-Streamlit tests passed.")


# ---- HINTS (uncover only if stuck > 5 min) ----
# Scaffold 3, Step 2: compare the stored date against today — what changes day-to-day?
# Scaffold 4, guest: compare count against the constant at the top of the file
# Scaffold 5, Step 1: same .get() pattern you used in check_password — what key?


# ---- Reflection questions ----
# Q1: _guest_count() uses a file on disk (/tmp/). What happens if BEKE runs
#     on App Runner with multiple container instances? Would guests get 3
#     queries per container or 3 total? What would you use instead?
#
# Q2: login_page() uses st.form() instead of bare st.text_input(). Besides
#     batching, what UX problem does the form solve for password fields?

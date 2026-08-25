"""Simple standalone test for the Oracle Fusion AI Agent Studio API.

Edit the four values below and run:  python test_oracle_auth.py

It sends one invokeAsync request and prints exactly what Oracle returns, so
you can see whether the credentials work (jobId) or are rejected (401), and
whether Basic auth is accepted at all. No other project files are needed —
just `pip install requests` if you don't already have it.
"""

import base64
import json

import requests  # pip install requests

# ---------------------------------------------------------------------------
# 1) EDIT THESE FOUR VALUES
# ---------------------------------------------------------------------------
BASE_URL   = "https://ejfh-dev4.fa.us6.oraclecloud.com"
AGENT_CODE = "HR_POLICY_WORKFLOW_AGENT_V35_WITHOUT_CHATSTORE"
USERNAME   = "your.username"          # <-- your Fusion username
PASSWORD   = "your.password"          # <-- your Fusion password
# ---------------------------------------------------------------------------

url = f"{BASE_URL}/api/fusion-ai/orchestrator/agent/v2/{AGENT_CODE}/invokeAsync"

# Build a Basic auth header from username:password
token = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
headers = {
    "Authorization": f"Basic {token}",
    "Content-Type": "application/json",
}

body = {
    "message": "Hello, this is a connection test.",
    "conversational": True,
    "conversationId": None,
    "invocationMode": "END_USER",
    "parameters": {},
}

print(f"POST {url}\n")

try:
    resp = requests.post(url, headers=headers, json=body, timeout=60)
except Exception as exc:
    print(f"Could not reach the server at all: {exc}")
    raise SystemExit(1)

print(f"HTTP status: {resp.status_code}\n")

# Try to pretty-print JSON; fall back to raw text.
try:
    print("Response body:")
    print(json.dumps(resp.json(), indent=2))
except ValueError:
    print("Response text:")
    print(resp.text[:2000])

print()
if resp.status_code == 200:
    print("✅ SUCCESS — Basic auth works. Look for a 'jobId' above.")
elif resp.status_code == 401:
    print("❌ 401 UNAUTHORIZED — credentials rejected. Either the username/password "
          "is wrong, or this API needs an OAuth2 Bearer token instead of Basic auth.")
elif resp.status_code == 403:
    print("❌ 403 FORBIDDEN — auth accepted, but this user lacks permission to invoke "
          "the agent. Ask an admin to grant the AI Agent Studio invocation role.")
elif resp.status_code == 404:
    print("❌ 404 NOT FOUND — check the BASE_URL and AGENT_CODE are exactly right.")
else:
    print(f"⚠️  Unexpected status {resp.status_code} — see the body above.")

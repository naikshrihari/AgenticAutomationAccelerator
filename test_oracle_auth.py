"""Standalone test for the Oracle Fusion AI Agent Studio API.

Edit the values below and run:  python test_oracle_auth.py

Supports three auth modes — set AUTH_MODE to whichever you can use:

  "basic"   -> username + password (only works for local, non-SSO accounts)
  "token"   -> paste a Bearer token you already have
  "oauth2"  -> fetch a Bearer token from IDCS/IAM using client credentials

It sends one invokeAsync request and prints exactly what Oracle returns, so you
can see whether auth works (jobId) or is rejected (401). Needs only `requests`.
"""

import base64
import json

import requests  # pip install requests

# ---------------------------------------------------------------------------
# COMMON — always fill these in
# ---------------------------------------------------------------------------
BASE_URL   = "https://ejfh-dev4.fa.us6.oraclecloud.com"
AGENT_CODE = "HR_POLICY_WORKFLOW_AGENT_V35_WITHOUT_CHATSTORE"

# Which auth to use: "basic" | "token" | "oauth2"
AUTH_MODE  = "basic"

# --- for AUTH_MODE = "basic" ---
USERNAME   = "your.username"
PASSWORD   = "your.password"

# --- for AUTH_MODE = "token" ---
BEARER_TOKEN = "paste-a-token-here"

# --- for AUTH_MODE = "oauth2" (fetch a token from IDCS/IAM) ---
TOKEN_URL     = "https://idcs-XXXX.identity.oraclecloud.com/oauth2/v1/token"
CLIENT_ID     = "your-client-id"
CLIENT_SECRET = "your-client-secret"
SCOPE         = "urn:opc:resource:consumer::all"   # ask your admin for the exact scope
# ---------------------------------------------------------------------------


def get_oauth2_token() -> str:
    """Client-credentials grant against IDCS/IAM; returns an access token."""
    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials", "scope": SCOPE},
        timeout=60,
    )
    print(f"Token endpoint HTTP {resp.status_code}")
    resp.raise_for_status()
    return resp.json()["access_token"]


# Build the Authorization header for the chosen mode.
if AUTH_MODE == "basic":
    token = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
    auth_header = f"Basic {token}"
elif AUTH_MODE == "token":
    auth_header = f"Bearer {BEARER_TOKEN}"
elif AUTH_MODE == "oauth2":
    auth_header = f"Bearer {get_oauth2_token()}"
else:
    raise SystemExit(f"Unknown AUTH_MODE: {AUTH_MODE!r}")

url = f"{BASE_URL}/api/fusion-ai/orchestrator/agent/v2/{AGENT_CODE}/invokeAsync"
headers = {"Authorization": auth_header, "Content-Type": "application/json"}
body = {
    "message": "Hello, this is a connection test.",
    "conversational": True,
    "conversationId": None,
    "invocationMode": "END_USER",
    "parameters": {},
}

print(f"\nAUTH_MODE={AUTH_MODE}")
print(f"POST {url}\n")

try:
    resp = requests.post(url, headers=headers, json=body, timeout=60)
except Exception as exc:
    print(f"Could not reach the server: {exc}")
    raise SystemExit(1)

print(f"HTTP status: {resp.status_code}\n")
try:
    print("Response body:")
    print(json.dumps(resp.json(), indent=2))
except ValueError:
    print("Response text:")
    print(resp.text[:2000])

print()
if resp.status_code == 200:
    print("SUCCESS - auth works. Look for a 'jobId' above.")
elif resp.status_code == 401:
    print("401 UNAUTHORIZED - this auth was rejected. If you used 'basic', the API "
          "likely needs a Bearer token: switch AUTH_MODE to 'token' or 'oauth2'.")
elif resp.status_code == 403:
    print("403 FORBIDDEN - auth accepted, but this identity lacks permission to "
          "invoke the agent. Ask an admin to grant the invocation role/scope.")
elif resp.status_code == 404:
    print("404 NOT FOUND - check BASE_URL and AGENT_CODE.")
else:
    print(f"Unexpected status {resp.status_code} - see body above.")

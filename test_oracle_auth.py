"""Full 3-step test for the Oracle Fusion AI Agent Studio API.

Runs the real flow and prints the COMPLETE responses so we can see exactly
where the answer text is (or why it's empty):

    1. get a bearer token   (paste one, or fetch via OAuth2 client creds)
    2. invokeAsync          -> jobId
    3. poll status          -> full JSON, including the 'answer' field

Edit the values below and run:  python test_oracle_auth.py
Needs only `requests`  (pip install requests).
"""

import base64
import json
import time

import requests

# ---------------------------------------------------------------------------
# FILL THESE IN
# ---------------------------------------------------------------------------
BASE_URL   = "https://ejfh-dev4.fa.us6.oraclecloud.com"
AGENT_CODE = "HR_POLICY_WORKFLOW_AGENT_V35_WITHOUT_CHATSTORE"
QUESTION   = "What is the sick leave policy?"

# How to get the bearer token: "token" (paste one) or "oauth2" (client creds)
AUTH_MODE  = "token"

# --- AUTH_MODE = "token": paste the access_token from the tokenrelay call ---
BEARER_TOKEN = "paste-access_token-here"

# --- AUTH_MODE = "oauth2" ---
TOKEN_URL     = "https://idcs-XXXX.identity.oraclecloud.com/oauth2/v1/token"
CLIENT_ID     = "your-client-id"
CLIENT_SECRET = "your-client-secret"
SCOPE         = "urn:opc:resource:consumer::all"

STATUS_INVOCATION_MODE = "ADMIN"    # the status poll uses ?invocationMode=ADMIN
# ---------------------------------------------------------------------------


def get_token() -> str:
    if AUTH_MODE == "token":
        return BEARER_TOKEN
    if AUTH_MODE == "oauth2":
        basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        r = requests.post(
            TOKEN_URL,
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials", "scope": SCOPE}, timeout=60)
        r.raise_for_status()
        return r.json()["access_token"]
    raise SystemExit(f"Unknown AUTH_MODE {AUTH_MODE!r}")


def show(title, resp):
    print(f"\n===== {title} — HTTP {resp.status_code} =====")
    try:
        print(json.dumps(resp.json(), indent=2)[:4000])
    except ValueError:
        print(resp.text[:2000])


token = get_token()
auth = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# --- Step 2: invokeAsync ---------------------------------------------------
invoke_url = f"{BASE_URL}/api/fusion-ai/orchestrator/agent/v2/{AGENT_CODE}/invokeAsync"
body = {"message": QUESTION, "conversational": True, "conversationId": None,
        "invocationMode": "END_USER", "parameters": {}}
inv = requests.post(invoke_url, headers=auth, json=body, timeout=60)
show("invokeAsync", inv)
if inv.status_code != 200:
    raise SystemExit("invokeAsync failed — see status above (401 = token issue).")

job_id = inv.json().get("jobId") or inv.json().get("id")
print(f"\njobId = {job_id}")
if not job_id:
    raise SystemExit("No jobId returned — cannot poll status.")

# --- Step 3: poll status ---------------------------------------------------
status_url = f"{BASE_URL}/api/fusion-ai/orchestrator/agent/v2/{AGENT_CODE}/status/{job_id}"
params = {"invocationMode": STATUS_INVOCATION_MODE}

final = None
for attempt in range(60):
    st = requests.get(status_url, headers=auth, params=params, timeout=60)
    data = st.json()
    state = str(data.get("status", "")).upper()
    print(f"poll {attempt + 1}: status = {state or '(none)'}")
    if state in {"COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED", "FAILED", "ERROR"}:
        final = st
        break
    time.sleep(2)

if final is None:
    raise SystemExit("Job never reached a terminal status (timed out).")

show("FINAL status", final)

# --- Where is the answer? --------------------------------------------------
data = final.json()
print("\n----- SUMMARY -----")
print("top-level keys:", list(data.keys()))
print("output field  :", repr(data.get("output")))   # Oracle puts the reply here
print("answer field  :", repr(data.get("answer")))
print("error field   :", repr(data.get("error")))
print("status        :", repr(data.get("status")))
print("\nThe agent reply is in 'output'. If it is null with status RUNNING, the "
      "job needs more time — increase the poll loop.")

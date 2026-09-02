# AgentProbe — AI Agent Test Automation Accelerator

A Python pipeline that reads your source documents, **generates a graded test
set**, runs it against a **target AI agent** through a platform connector, and
**reports** the results. Every language-model step — question generation,
expected-answer generation, semantic grading, and embeddings — runs locally on
**[Ollama](https://ollama.com)**. Because inference is on-premise, document and
HR data never leave your network, which suits regulated HR and gaming data.

> Author: Shrihari, AI Architect, SmarTek21 · Status: v1.0

## Architecture

```
 INPUTS
 ├── Documents folder   (PDF, DOCX, HTML, MD, TXT — the source of truth)
 ├── Target config      (one YAML per agent: endpoint, auth, mapping, thresholds)
 └── Seed cases         (optional curated adversarial / negative cases)
        │
        ▼
 1. Ingestion & Processing      loaders → cleaner → section-aware chunker → metadata tagger
        │
        ▼
 2. Generation (Ollama)         question + grounded answer + facts + citation + type + difficulty
        │                        self-consistency pass · embedding dedup + semantic index
        ▼
 3. Golden Set (versioned)      approved cases as versioned JSONL with citations · human review gate
        │
        ▼
 4. Execution Engine            async runner (retries, sessions) via the connector layer
        │                        Oracle Fusion · ServiceNow · Generic REST · OpenAI-compatible
        ▼                        infra errors classified separately from wrong answers
 5. Grading (Ollama)            fact-coverage checklist · LLM-as-judge (rubric) · groundedness
        │                        verdict resolver → pass / partial / fail + score + rationale
        ▼
 6. Reporting                   results store (DuckDB/SQLite) · regression diff vs baseline
        │                        HTML report · CSV export · push to native evaluators
        ▼
 OUTPUTS  →  HTML report · CSV export · Oracle Evaluation Sets / ServiceNow Agentic Evaluations
```

The same flow runs against the next platform by changing **only the connector
and config** — nothing in the core is platform-specific.

## Root-cause analysis (why did it fail?)

Beyond pass/fail, AgentProbe can explain **why** cases failed. Enable it with the
"🔎 Analyze failures" checkbox in the UI's Evaluate tab, or `--analyze` on the
CLI. It embeds each failure, clusters similar ones together, and asks the local
judge model to diagnose each cluster — producing a **category, likely cause, and
a concrete suggested fix** — which appears as a "Why did it fail?" section in the
HTML report. For example, several failures might be grouped as
*"wrong_source_retrieved — the agent returns an adjacent handbook section; add
section titles to the embedding text."* All of it runs locally on Ollama.

## Security & Safety Tests (adversarial)

Generate safety & compliance probes — **PII leakage, prompt injection, jailbreak,
unauthorized actions, and bias** — where a correct agent *refuses* or escalates.
Use the "🛡️ Security & Safety" tab in the UI, or the CLI:

```bash
agentprobe redteam --domain "an HR policy assistant" --variants 3
```

It saves an adversarial golden set (graded on whether the agent declines, not on
fact coverage), which you then run in the Evaluate tab like any other set. Pair
it with `--analyze` to diagnose *why* the agent leaked or complied.

## History & trends

Every evaluation is stored locally, and the UI's "④ History & trends" tab charts
**pass rate over time** per target and lists past runs — so regressions and
improvements across runs are visible at a glance.

## Why Ollama

- **On-premise inference** — documents and HR data stay inside the network.
- **No per-token cost** — large suites and frequent re-runs are affordable.
- **Model choice per task** — a stronger model for judging, a lighter one for
  generation, a dedicated embedding model, all swappable by name.
- **Provider-agnostic core** — the engine calls Ollama through an
  OpenAI-compatible interface, so a hosted model can be substituted later
  without changing the pipeline.

## Install

```bash
pip install -e ".[full]"     # core + all optional parsers/reporting/store
# or minimal:
pip install -e .             # pydantic + httpx + pyyaml only
```

Pull the models on your Ollama host:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Configure

Settings come from the environment (all optional; sensible defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `AGENTPROBE_GEN_MODEL` | `llama3.1` | Generation model |
| `AGENTPROBE_JUDGE_MODEL` | `llama3.1` | Judge model (use a stronger one if you have it) |
| `AGENTPROBE_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `AGENTPROBE_DATA_DIR` | `data` | Where golden sets, results, and reports are written |

Each target agent is one YAML file — see [`config/`](config/) for Oracle
Fusion, ServiceNow, and OpenAI-compatible examples.

### Oracle Fusion AI Agent Studio

The Oracle connector implements Oracle's real **async** contract — POST
`…/orchestrator/agent/v2/{agentCode}/invokeAsync` to get a `jobId`, then poll
`…/status/{jobId}` until the `answer` is ready. To point it at your agent, edit
`config/oracle_fusion.yaml` and set just three things:

1. `base_url` — your Fusion host
2. `options.agent_code` — your AI agent (team) code / name
3. credentials — set `ORACLE_FUSION_USER` / `ORACLE_FUSION_PASSWORD` (Basic auth),
   or switch to a bearer token

No code changes are needed — session continuity, polling, retries, and error
classification are handled by the connector.

## Use — Web UI (no command line)

Prefer a browser to the terminal? There's a Streamlit app that does everything —
upload documents, generate the questions with a live progress bar, browse and
download the golden set, and run an evaluation with an inline report.

```bash
pip install -e ".[ui]"     # or: pip install streamlit
streamlit run app.py       # opens http://localhost:8501 in your browser
```

On **Windows** you can just **double-click `run_ui.bat`** — it activates the
virtual environment and launches the UI for you.

The UI has three tabs: **① Generate questions** (upload a PDF, pick question
types / chunk size / limit, click Generate), **② Browse golden sets** (view and
download any saved set as CSV), and **③ Evaluate an agent** (run a golden set
against a target config and view the HTML report inline). Model and Ollama
settings are in the sidebar, with a one-click "Test Ollama connection" button.

### Optional: steer generation with a business-requirements file

Both the UI and CLI accept an **optional business-requirements document** (BRD).
When provided, the generator prioritises questions that verify those
requirements wherever a source section is relevant — while keeping every
expected answer grounded strictly in the source documents. Omit it and
generation proceeds from the documents alone.

- **UI:** use the "Business requirements (optional)" uploader on the Generate tab.
- **CLI:** `agentprobe generate --docs .\documents --requirements .\brd.pdf`

The BRD can be any supported format (PDF, DOCX, HTML, Markdown, TXT).

## Use — Command line

```bash
# 1) Build a versioned golden set from your documents (+ optional seed cases)
agentprobe generate --docs ./examples/documents --seeds ./examples/seed_cases.jsonl

# 2) Run the latest golden set against a target and produce a report
agentprobe evaluate --target config/oracle_fusion.yaml --fail-under 0.8

# Or do both at once
agentprobe all --docs ./examples/documents --target config/openai_compat.yaml
```

Python:

```python
from agentprobe import Pipeline, TargetConfig

pipe = Pipeline()
pipe.build_golden_set("./examples/documents")           # ingest + generate + store
summary = pipe.evaluate(TargetConfig.from_yaml("config/oracle_fusion.yaml"))
print(summary.pass_rate)                                 # excludes infra errors
```

Outputs land under `data/reports/<run_id>/`: `report.html`, `results.csv`, and
`eval_set.json` (for native evaluators). Runs are stored so the next run is
automatically **diffed against the previous baseline** and flags any case that
regressed.

## Package layout

```
agentprobe/
├── config.py            Settings + per-target YAML config
├── models.py            Pydantic models exchanged between stages
├── llm/                 Ollama client (OpenAI-compatible) + embeddings
├── ingestion/           loaders · cleaner · section-aware chunker · tagger
├── generation/          generator · self-consistency · dedup + semantic index
├── golden/              versioned JSONL store · human review gate
├── execution/           async runner + connectors/ (oracle_fusion, servicenow, generic_rest, openai_compat)
├── grading/             fact-coverage · LLM judge · groundedness · verdict resolver
├── reporting/           results store · regression differ · HTML/CSV/native report builder
├── pipeline.py          end-to-end orchestration
└── cli.py               generate / evaluate / all
```

## Technology stack

Python 3.11+ · Ollama (generation, judge, embedding models) · Pydantic for typed
cases and verdicts · `asyncio` + `httpx` + `tenacity` for parallel runs and
retries · versioned JSONL golden set · DuckDB/SQLite results store · Jinja2 HTML
report + CSV · YAML per target.

Optional dependencies degrade gracefully: without `tenacity` a manual backoff
loop is used, without `jinja2` a plain-HTML report is emitted, and without
`duckdb` the store falls back to stdlib `sqlite3`.

## Tests

```bash
pytest        # offline tests: ingestion, grading checks, resolver, differ, store
```

The offline suite needs no Ollama server; LLM-backed stages are exercised by
integration tests against a live Ollama instance.

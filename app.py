"""AgentProbe web UI (Streamlit).

A local, no-terminal front end for the accelerator. Upload one or more source
documents, generate the graded question / expected-answer set on Ollama, browse
and download it, and (optionally) run it against a target agent and view the
report — all from the browser.

Run it with:

    streamlit run app.py

Everything still runs locally: the UI just drives the same pipeline the CLI
uses, so all LLM inference goes to your local Ollama server.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Load .env so the sidebar defaults and any credentials come from it.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agentprobe.config import Settings, TargetConfig
from agentprobe.golden import GoldenSetStore, ReviewGate
from agentprobe.generation import CaseGenerator, Deduplicator, SelfConsistencyChecker
from agentprobe.ingestion import chunk_document, clean_text, tag_chunks
from agentprobe.ingestion.loaders import SUPPORTED_SUFFIXES, load_document
from agentprobe.llm import EmbeddingModel, OllamaClient
from agentprobe.models import QuestionType

st.set_page_config(page_title="AgentProbe", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_settings() -> Settings:
    """Read the sidebar controls into a Settings object."""
    return Settings(
        ollama_base_url=st.session_state.ollama_url,
        generation_model=st.session_state.gen_model,
        judge_model=st.session_state.judge_model,
        embedding_model=st.session_state.embed_model,
        request_timeout_s=float(os.environ.get("AGENTPROBE_TIMEOUT", 600)),
        data_dir=Path(st.session_state.data_dir),
    )


def check_ollama(settings: Settings) -> tuple[bool, str]:
    """Ping the Ollama server and list installed models."""
    import httpx

    # /v1/models is the OpenAI-compatible listing endpoint.
    url = settings.ollama_base_url.rstrip("/") + "/models"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        names = [m.get("id", "?") for m in resp.json().get("data", [])]
        return True, ", ".join(names) or "(no models installed)"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def save_uploads_to_tmp(files) -> list[Path]:
    """Copy uploaded files to a temp folder (main thread) and return their paths.

    Must run on the main thread because Streamlit's uploaded-file objects are
    bound to the session; the heavy parsing then happens in a worker thread.
    """
    tmp = Path(tempfile.mkdtemp(prefix="agentprobe_upload_"))
    paths: list[Path] = []
    for uploaded in files:
        dest = tmp / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        paths.append(dest)
    return paths


def chunks_from_paths(paths: list[Path], chunk_size: int):
    """Load, clean, chunk, and tag documents from paths (worker-thread safe)."""
    chunks = []
    for p in paths:
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        doc = load_document(p)
        doc.text = clean_text(doc.text)
        chunks.extend(chunk_document(doc, max_tokens=chunk_size))
    return tag_chunks(chunks)


# --------------------------------------------------------------------------- #
# Background jobs — run long tasks off the Streamlit script thread so that
# changing a widget (which reruns the script) never interrupts them.
# --------------------------------------------------------------------------- #
class JobCancelled(Exception):
    """Raised inside a worker when the user requests cancellation."""


def new_job() -> dict:
    return {"status": "running", "phase": "starting", "exec_done": 0,
            "grade_done": 0, "gen_done": 0, "total": 0, "rows": [],
            "result": None, "error": None, "message": "", "cancel": False}


def start_job(fn) -> None:
    """Run fn(job-mutating) in a daemon thread. fn sets job['status']."""
    threading.Thread(target=fn, daemon=True).start()


def extract_requirements(req_file) -> str | None:
    """Read an optional uploaded business-requirements file into plain text."""
    if req_file is None:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="agentprobe_req_"))
    dest = tmp / req_file.name
    dest.write_bytes(req_file.getbuffer())
    if dest.suffix.lower() not in SUPPORTED_SUFFIXES:
        st.warning(f"Unsupported requirements file type: {req_file.name} — ignored.")
        return None
    return clean_text(load_document(dest).text)


def _read_rows(uploaded) -> list[dict]:
    """Read an uploaded CSV or Excel file into a list of {header: value} rows."""
    import io as _io

    name = (uploaded.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        import pandas as _pd  # needs openpyxl for .xlsx

        df = _pd.read_excel(_io.BytesIO(uploaded.getvalue()), dtype=str).fillna("")
        return df.to_dict(orient="records")
    # CSV
    import csv as _csv

    text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
    return list(_csv.DictReader(_io.StringIO(text)))


def cases_from_upload(uploaded) -> list:
    """Build TestCase objects from an uploaded CSV or Excel file.

    Accepts flexible column names for question / expected_answer, and optional
    'required_facts' (';'-separated) and 'question_type' columns.
    """
    import uuid as _uuid

    from agentprobe.models import QuestionType as _QT, TestCase as _TC

    rows = _read_rows(uploaded)
    if not rows:
        return []
    # Normalise headers so "Expected Answer", "expected_answer", "answer" all work.
    field_map = {(str(h) or "").strip().lower(): h for h in rows[0].keys()}

    def pick(row, *names):
        for n in names:
            key = field_map.get(n)
            if key is not None and str(row.get(key, "")).strip():
                return str(row[key]).strip()
        return ""

    cases = []
    for row in rows:
        question = pick(row, "question", "q", "prompt")
        expected = pick(row, "expected_answer", "expected answer", "expected", "answer", "ground_truth")
        if not question:
            continue  # skip blank/malformed rows
        facts = pick(row, "required_facts", "required facts", "facts")
        qtype = pick(row, "question_type", "type").lower()
        cases.append(_TC(
            case_id=f"up-{_uuid.uuid4().hex[:10]}",
            question=question,
            expected_answer=expected,
            required_facts=[f.strip() for f in facts.split(";") if f.strip()],
            question_type=_QT(qtype) if qtype in {t.value for t in _QT} else _QT.FACTUAL,
            approved=True,
        ))
    return cases


def cases_to_records(cases) -> list[dict]:
    """Flatten test cases into rows for a table / CSV.

    Includes an attack_category column only when at least one case is a red-team
    case, so ordinary golden sets aren't cluttered with an empty column.
    """
    has_adversarial = any(getattr(c, "attack_category", None) for c in cases)
    rows = []
    for c in cases:
        row = {
            "question": c.question,
            "expected_answer": c.expected_answer,
            "type": c.question_type.value,
            "difficulty": c.difficulty.value,
            "language": c.language,
        }
        if has_adversarial:
            row["attack_category"] = getattr(c, "attack_category", None) or ""
        row["citation"] = c.citation
        row["required_facts"] = " | ".join(c.required_facts)
        rows.append(row)
    return rows


def records_to_csv(records: list[dict]) -> bytes:
    import csv

    buf = io.StringIO()
    if not records:
        return b""
    writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------- #
# Sidebar — settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("Ollama URL", value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), key="ollama_url")
    st.text_input("Generation model", value=os.environ.get("AGENTPROBE_GEN_MODEL", "llama3.2:1b"), key="gen_model")
    st.text_input("Judge model", value=os.environ.get("AGENTPROBE_JUDGE_MODEL", "llama3.2:1b"), key="judge_model")
    st.text_input("Embedding model", value=os.environ.get("AGENTPROBE_EMBED_MODEL", "nomic-embed-text"), key="embed_model")
    st.text_input("Data folder", value=os.environ.get("AGENTPROBE_DATA_DIR", "data"), key="data_dir")

    if st.button("🔌 Test Ollama connection", width='stretch'):
        ok, detail = check_ollama(build_settings())
        if ok:
            st.success(f"Connected. Models: {detail}")
        else:
            st.error(f"Cannot reach Ollama: {detail}")

    st.caption("All inference runs locally on your Ollama server. "
               "Smaller models (e.g. llama3.2:1b) are much faster on CPU.")


# --------------------------------------------------------------------------- #
# Main — tabs
# --------------------------------------------------------------------------- #
st.title("🧪 AgentProbe")
st.caption("AI Agent Test Automation Accelerator — generate a graded test set from your "
           "documents and evaluate an agent, all on local Ollama.")

tab_generate, tab_redteam, tab_browse, tab_evaluate, tab_history = st.tabs(
    ["① Generate questions", "🛡️ Security & Safety", "② Browse evaluation sets",
     "③ Evaluate an agent", "④ History & trends"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — Generate
# --------------------------------------------------------------------------- #
with tab_generate:
    st.subheader("Generate questions & expected answers")
    st.write("Upload one or more documents (PDF, DOCX, HTML, Markdown, TXT). "
             "Each section becomes graded test cases.")

    files = st.file_uploader(
        "Source documents",
        type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
        accept_multiple_files=True,
    )

    req_file = st.file_uploader(
        "Business requirements (optional)",
        type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
        accept_multiple_files=False,
        help="Optional. If provided, questions are steered toward verifying these "
             "business requirements. Expected answers stay grounded in the source documents.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        types = st.multiselect(
            "Question types",
            options=[t.value for t in QuestionType],
            default=["factual"],
            help="More types = more questions per section, but more model calls.",
        )
    with col2:
        chunk_size = st.slider("Chunk size (tokens)", 150, 800, 400, step=50,
                               help="Smaller chunks = more sections = more questions.")
        limit = st.number_input("Limit sections (0 = all)", min_value=0, value=5, step=1,
                                help="Cap how many sections to use — keep small for a quick trial.")
    with col3:
        workers = st.slider("Parallel workers", 1, 8, 2,
                            help="Generate several sections at once (needs a capable Ollama box).")
        do_validate = st.checkbox("Self-consistency check", value=False,
                                  help="Extra Ollama pass to verify answers are grounded. Slower.")
        do_dedupe = st.checkbox("De-duplicate", value=False,
                                help="Embedding-based removal of near-duplicate questions.")

    gen_job = st.session_state.get("gen_job")
    gen_running = bool(gen_job and gen_job["status"] == "running")

    if st.button("🚀 Generate", type="primary", disabled=not files or gen_running):
        settings = build_settings()
        ok, detail = check_ollama(settings)
        if not ok:
            st.error(f"Ollama is not reachable at {settings.ollama_base_url}. "
                     f"Start it and pull your models first.\n\n{detail}")
            st.stop()
        if not types:
            st.error("Pick at least one question type.")
            st.stop()

        # Clear the previous result so a stale table isn't shown during the run.
        st.session_state.pop("gen_last_cases", None)

        # Save uploads on the main thread; parse/generate in the worker thread.
        doc_paths = save_uploads_to_tmp(files)
        req_path = save_uploads_to_tmp([req_file])[0] if req_file else None
        job = new_job()
        st.session_state["gen_job"] = job
        _types = [QuestionType(t) for t in types]
        _limit, _chunk_size, _workers = int(limit), int(chunk_size), int(workers)
        _do_validate, _do_dedupe = do_validate, do_dedupe

        def _run_generation():
            try:
                job["phase"] = "ingesting"
                chunks = chunks_from_paths(doc_paths, _chunk_size)
                if _limit:
                    chunks = chunks[:_limit]
                if not chunks:
                    job["status"] = "error"
                    job["error"] = "No readable text extracted (a scanned PDF needs OCR first)."
                    return
                job["total"] = len(chunks)
                requirements = clean_text(load_document(req_path).text) if req_path else None
                generator = CaseGenerator(
                    client=OllamaClient(settings, model=settings.generation_model),
                    settings=settings, type_mix=_types, max_workers=_workers,
                    requirements=requirements)
                job["phase"] = "generating"
                cases = []
                for i, chunk in enumerate(chunks, start=1):
                    if job["cancel"]:
                        job["status"] = "cancelled"
                        return
                    cases.extend(generator.generate_for_chunk(chunk))
                    job["gen_done"] = i
                    job["message"] = f"Section {i}/{len(chunks)} — {len(cases)} questions so far"
                if _do_validate and cases:
                    job["phase"] = "validating"
                    checker = SelfConsistencyChecker(settings=settings)
                    cases = checker.filter(cases, {c.chunk_id: c for c in chunks})
                if _do_dedupe and cases:
                    job["phase"] = "deduping"
                    cases = Deduplicator(EmbeddingModel(settings), settings).dedupe(cases)
                cases = ReviewGate().run(cases)
                saved_path = GoldenSetStore(settings.golden_dir).save_version(cases)
                job["result"] = {"cases": cases, "path": str(saved_path)}
                job["status"] = "done"
            except Exception as exc:  # noqa: BLE001
                job["status"] = "error"
                job["error"] = str(exc)

        start_job(_run_generation)
        st.rerun()

    # Progress / result area for generation (polls while the thread runs).
    if gen_job:
        if gen_job["status"] == "running":
            total = gen_job["total"] or 1
            frac = (gen_job["gen_done"] / total) if gen_job["phase"] == "generating" else 0.02
            st.progress(min(frac, 0.99), text=f"{gen_job['phase']} — {gen_job['message']}")
            st.caption("Running in the background — you can change settings without stopping it.")
            if gen_job["cancel"]:
                st.warning("Cancelling — stopping after the current section…")
            elif st.button("🛑 Cancel", key="gen_cancel"):
                gen_job["cancel"] = True
                st.rerun()
            time.sleep(0.7)
            st.rerun()
        elif gen_job["status"] == "cancelled":
            st.warning("Generation cancelled.")
            if st.button("Clear", key="gen_clear_cancel"):
                del st.session_state["gen_job"]; st.rerun()
        elif gen_job["status"] == "error":
            st.error(f"Generation failed: {gen_job['error']}")
            if st.button("Clear", key="gen_clear_err"):
                del st.session_state["gen_job"]; st.rerun()
        elif gen_job["status"] == "done":
            res = gen_job["result"]
            st.session_state["gen_last_cases"] = res["cases"]
            st.success(f"Saved evaluation set with {len(res['cases'])} questions to `{res['path']}`")
            del st.session_state["gen_job"]

    # Show this tab's most recent result (its own slot, not shared with the
    # Security & Safety tab).
    if st.session_state.get("gen_last_cases"):
        cases = st.session_state["gen_last_cases"]
        records = cases_to_records(cases)
        st.markdown(f"### Generated questions ({len(records)})")
        st.dataframe(records, width='stretch', height=400)

        jsonl = "\n".join(c.model_dump_json() for c in cases).encode("utf-8")
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download CSV", records_to_csv(records),
                           file_name="questions.csv", mime="text/csv", width='stretch')
        c2.download_button("⬇️ Download JSONL", jsonl,
                           file_name="golden_set.jsonl", mime="application/json",
                           width='stretch')


# --------------------------------------------------------------------------- #
# Tab — Red-team (adversarial safety / compliance cases)
# --------------------------------------------------------------------------- #
with tab_redteam:
    st.subheader("🛡️ Security & Safety Tests")
    st.write("Generate safety & compliance probes — PII leakage, prompt injection, "
             "jailbreak, unauthorized actions, bias — where a correct agent **refuses** "
             "or escalates. Saved as an evaluation set you can run in the Evaluate tab.")

    from agentprobe.adversarial import ATTACK_CATEGORIES

    rt_domain = st.text_input("Agent domain (seeds realistic variants)",
                              value="a company HR policy assistant")
    cat_labels = {c.key: c.title for c in ATTACK_CATEGORIES}
    rt_cats = st.multiselect("Attack categories", options=list(cat_labels.keys()),
                             default=list(cat_labels.keys()),
                             format_func=lambda k: cat_labels[k])
    rt_variants = st.slider("Extra LLM-generated prompts per category", 0, 5, 0,
                            help="0 = curated attacks only (instant). Higher = more "
                                 "variety, but calls the local model.")

    if st.button("🛡️ Generate security & safety tests", type="primary", disabled=not rt_cats):
        from agentprobe.pipeline import Pipeline

        settings = build_settings()
        with st.spinner("Generating security & safety tests…"):
            try:
                rt_cases = Pipeline(settings).build_redteam_set(
                    domain=rt_domain, categories=rt_cats,
                    llm_variants_per_category=int(rt_variants))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Security & safety test generation failed: {exc}")
                st.stop()
        st.success(f"Generated **{len(rt_cases)}** security & safety test cases and saved them as an evaluation set.")
        by_cat = {}
        for c in rt_cases:
            by_cat[c.attack_category] = by_cat.get(c.attack_category, 0) + 1
        st.write("By category:", by_cat)
        st.dataframe([{"category": c.attack_category, "attack": c.question} for c in rt_cases],
                     width='stretch', height=320)


# --------------------------------------------------------------------------- #
# Tab 2 — Browse existing golden sets
# --------------------------------------------------------------------------- #
with tab_browse:
    st.subheader("Browse saved evaluation sets")
    settings = build_settings()
    store = GoldenSetStore(settings.golden_dir)
    versions = store.versions()
    if not versions:
        st.info(f"No evaluation sets found under `{settings.golden_dir}`. Generate one first.")
    else:
        choice = st.selectbox("Evaluation set version", options=list(reversed(versions)))
        if choice:
            cases = store.load(settings.golden_dir / choice)
            records = cases_to_records(cases)
            st.markdown(f"**{len(records)}** questions in `{choice}`")
            st.dataframe(records, width='stretch', height=420)
            st.download_button("⬇️ Download CSV", records_to_csv(records),
                               file_name=f"{choice}.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# Tab 3 — Evaluate an agent
# --------------------------------------------------------------------------- #
with tab_evaluate:
    st.subheader("Evaluate an agent against an evaluation set")
    st.write("Point a target config at your agent, then run the latest evaluation set against it. "
             "Grading (fact coverage + LLM judge) runs on Ollama.")

    config_files = sorted(str(p) for p in Path("config").glob("*.yaml")) if Path("config").exists() else []
    if not config_files:
        st.info("No target configs found in `config/`. Add a YAML describing your agent's endpoint.")
    else:
        target_path = st.selectbox("Target config", options=config_files)
        settings = build_settings()
        versions = GoldenSetStore(settings.golden_dir).versions()

        st.markdown("#### Questions to run")
        case_source = st.radio(
            "Question set",
            ["Saved evaluation set", "Upload CSV/Excel"],
            horizontal=True,
            help="Use an evaluation set generated in tab ①, or upload your own CSV with "
                 "question and expected_answer columns.",
        )
        golden_choice = None
        csv_upload = None
        mark_adversarial = False
        if case_source == "Saved evaluation set":
            golden_choice = st.selectbox("Evaluation set", options=list(reversed(versions)) or ["(none — generate first)"])
            st.caption("Tip: a security & safety test set generated in the 🛡️ tab appears here and is "
                       "already graded on refusal.")
        else:
            csv_upload = st.file_uploader(
                "CSV or Excel with 'question' and 'expected_answer' columns",
                type=["csv", "xlsx", "xls"])
            st.caption("Required columns: question, expected_answer. Optional: "
                       "required_facts (';'-separated), question_type.")
            mark_adversarial = st.checkbox(
                "🛡️ Treat these as security & safety tests (grade on refusal)",
                value=False,
                help="Check this when your uploaded questions are attacks (PII, "
                     "injection, jailbreak…). The agent passes by refusing, not by "
                     "answering. Overrides the question_type column.")

        # Load the chosen config so we can pre-fill the connection fields.
        base_target = TargetConfig.from_yaml(target_path)

        st.markdown("#### Connection")
        st.caption("Enter the agent's connection details here — they override the "
                   "config file at runtime and are not saved to disk.")
        ov_base_url = st.text_input("Base URL", value=base_target.base_url)

        is_oracle = base_target.connector == "oracle_fusion"
        ov_agent_code = ""
        ora_auth_mode = "Bearer token (paste)"
        ov_bearer = ov_cookie = ov_xsrf = ""
        ov_username = ov_password = ov_token = ""

        if is_oracle:
            existing_options = getattr(base_target, "options", {}) or {}
            ov_agent_code = st.text_input(
                "AI agent code / name",
                value=str(existing_options.get("agent_code", "")),
            )
            ora_auth_mode = st.radio(
                "Authentication",
                ["Bearer token (paste)", "Session cookie relay"],
                help="Oracle AI Agent Studio needs a bearer token. Paste one you "
                     "copied from the token-relay call, or paste your browser "
                     "session cookie + xsrf token and the tool will mint one.",
            )
            if ora_auth_mode == "Bearer token (paste)":
                ov_bearer = st.text_area(
                    "Bearer token (access_token from the tokenrelay call)", height=90)
            else:
                ov_cookie = st.text_area(
                    "Session cookie (the full Cookie header from your browser)", height=120)
                ov_xsrf = st.text_input("x-xsrf-token", value="")
                st.caption("These are sensitive session credentials and expire — "
                           "they are used only for this run and never saved to disk.")
        else:
            auth_type = base_target.auth.type
            cc1, cc2 = st.columns(2)
            if auth_type == "basic":
                ov_username = cc1.text_input("Username", value="")
                ov_password = cc2.text_input("Password", value="", type="password")
            elif auth_type in ("bearer", "api_key"):
                ov_token = st.text_input("Token", value="", type="password")
            else:
                st.caption(f"Auth type '{auth_type}' — no credentials needed.")

        analyze_failures = st.checkbox(
            "🔎 Analyze failures (root cause + suggested fixes)", value=False,
            help="After grading, cluster the failures and diagnose each group on the "
                 "local model, with a suggested fix. Adds a few extra model calls.")

        eval_job = st.session_state.get("eval_job")
        eval_running = bool(eval_job and eval_job["status"] == "running")

        can_run = bool(golden_choice and versions) if case_source == "Saved evaluation set" else (csv_upload is not None)  # noqa: E501
        if st.button("▶️ Run evaluation", type="primary", disabled=not can_run or eval_running):
            from agentprobe.pipeline import Pipeline

            # Apply the UI overrides onto the loaded config before running.
            target = base_target.model_copy(deep=True)
            if ov_base_url:
                target.base_url = ov_base_url
            if getattr(target, "options", None) is None:
                target.options = {}

            if is_oracle:
                if ov_agent_code:
                    target.options["agent_code"] = ov_agent_code
                if ora_auth_mode == "Bearer token (paste)":
                    target.options["auth_mode"] = "token"
                    target.auth.type = "bearer"
                    target.auth.token = ov_bearer.strip()
                    if not target.auth.token:
                        st.error("Please paste the bearer token.")
                        st.stop()
                else:
                    target.options["auth_mode"] = "relay"
                    target.options["relay_cookie"] = ov_cookie.strip()
                    target.options["relay_xsrf"] = ov_xsrf.strip()
                    if not (target.options["relay_cookie"] and target.options["relay_xsrf"]):
                        st.error("Please paste both the session cookie and the x-xsrf-token.")
                        st.stop()
            else:
                if ov_username:
                    target.auth.username = ov_username
                if ov_password:
                    target.auth.password = ov_password
                if ov_token:
                    target.auth.token = ov_token
                if base_target.auth.type == "basic" and not (target.auth.username and target.auth.password):
                    st.error("Please enter both username and password.")
                    st.stop()

            # Load the questions from the chosen source (main thread; fast).
            if case_source == "Upload CSV/Excel":
                try:
                    cases = cases_from_upload(csv_upload)
                except ImportError:
                    st.error("Reading Excel needs pandas + openpyxl. Install them with "
                             "`pip install pandas openpyxl`, or upload a CSV instead.")
                    st.stop()
                if not cases:
                    st.error("No usable rows found. The file needs a 'question' column "
                             "(and ideally 'expected_answer').")
                    st.stop()
                if mark_adversarial:
                    for c in cases:
                        c.question_type = QuestionType.ADVERSARIAL
                        if not (c.attack_category or ""):
                            c.attack_category = "uploaded"
            else:
                cases = GoldenSetStore(settings.golden_dir).load(settings.golden_dir / golden_choice)

            # Start the evaluation in a background thread so UI reruns can't stop it.
            job = new_job()
            job["total"] = len(cases)
            st.session_state["eval_job"] = job
            _analyze = analyze_failures

            def _run_eval():
                try:
                    def on_execute(done, total):
                        if job["cancel"]:
                            raise JobCancelled()
                        job["exec_done"] = done
                        job["phase"] = "querying"

                    def on_grade(done, total, result):
                        if job["cancel"]:
                            raise JobCancelled()
                        job["grade_done"] = done
                        job["phase"] = "grading"
                        job["rows"].append({
                            "case": result.case_id, "type": result.question_type.value,
                            "verdict": result.verdict.value, "score": round(result.score, 2)})

                    summary = Pipeline(settings).evaluate(
                        target, cases=cases, analyze_failures=_analyze,
                        on_execute=on_execute, on_grade=on_grade)
                    report = settings.reports_dir / summary.run_id / "report.html"
                    job["result"] = {
                        "summary": summary,
                        "report_html": report.read_text(encoding="utf-8") if report.exists() else None,
                        "report_path": str(report),
                    }
                    job["status"] = "done"
                except JobCancelled:
                    job["status"] = "cancelled"
                except Exception as exc:  # noqa: BLE001
                    job["status"] = "error"
                    job["error"] = str(exc)

            start_job(_run_eval)
            st.rerun()

        # Progress / result area for evaluation (polls while the thread runs).
        if eval_job:
            total = eval_job["total"] or 1
            if eval_job["status"] == "running":
                if eval_job["phase"] == "grading":
                    frac = 0.4 + 0.6 * eval_job["grade_done"] / total
                    label = f"Grading — {eval_job['grade_done']}/{total}"
                else:
                    frac = min(0.4, 0.4 * eval_job["exec_done"] / total)
                    label = f"Querying agent — {eval_job['exec_done']}/{total}"
                st.progress(min(frac, 0.99), text=label)
                st.caption("Running in the background — you can change settings without stopping it.")
                if eval_job["cancel"]:
                    st.warning("Cancelling — stopping after the current case…")
                elif st.button("🛑 Cancel", key="eval_cancel"):
                    eval_job["cancel"] = True
                    st.rerun()
                if eval_job["rows"]:
                    st.dataframe(eval_job["rows"], width='stretch', height=240)
                time.sleep(0.7)
                st.rerun()
            elif eval_job["status"] == "cancelled":
                st.warning("Evaluation cancelled (partial results were not saved).")
                if st.button("Clear", key="eval_clear_cancel"):
                    del st.session_state["eval_job"]; st.rerun()
            elif eval_job["status"] == "error":
                st.error(f"Evaluation failed: {eval_job['error']}")
                if st.button("Clear", key="eval_clear_err"):
                    del st.session_state["eval_job"]; st.rerun()
            elif eval_job["status"] == "done":
                res = eval_job["result"]
                summary = res["summary"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pass rate", f"{summary.pass_rate:.0%}")
                m2.metric("Passed", summary.passed)
                m3.metric("Failed", summary.failed)
                m4.metric("Infra errors", summary.errors)
                if res["report_html"]:
                    components.html(res["report_html"], height=600, scrolling=True)
                    st.download_button("⬇️ Download HTML report", res["report_html"].encode("utf-8"),
                                       file_name="report.html", mime="text/html")
                if st.button("Run another", key="eval_clear_done"):
                    del st.session_state["eval_job"]; st.rerun()


# --------------------------------------------------------------------------- #
# Tab — History & trends
# --------------------------------------------------------------------------- #
with tab_history:
    st.subheader("④ History & trends")
    st.write("Every evaluation is stored locally. Track how an agent's pass rate "
             "moves over time and spot regressions across runs.")

    settings = build_settings()
    from agentprobe.reporting.store import ResultsStore

    db_path = settings.results_dir / "results.db"
    if not db_path.exists():
        st.info("No runs recorded yet. Run an evaluation first (tab ③).")
    else:
        store = ResultsStore(db_path)
        targets = store.targets()
        pick = st.selectbox("Target", options=["(all)"] + targets)
        runs = store.list_runs(None if pick == "(all)" else pick)
        store.close()

        if not runs:
            st.info("No runs for this target yet.")
        else:
            # Trend chart: pass rate over time (oldest → newest).
            try:
                import pandas as pd

                df = pd.DataFrame(runs)
                df["started_at"] = pd.to_datetime(df["started_at"])
                df = df.sort_values("started_at")
                chart = df.pivot_table(index="started_at", columns="target",
                                       values="pass_rate", aggfunc="last")
                st.markdown("#### Pass rate over time")
                st.line_chart(chart)

                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else None
                m1, m2, m3 = st.columns(3)
                delta = (f"{(latest['pass_rate'] - prev['pass_rate']):+.0%}" if prev is not None else None)
                m1.metric("Latest pass rate", f"{latest['pass_rate']:.0%}", delta)
                m2.metric("Total runs", len(df))
                m3.metric("Avg pass rate", f"{df['pass_rate'].mean():.0%}")
            except ImportError:
                st.caption("Install pandas for the trend chart: pip install pandas")

            st.markdown("#### Run history")
            st.dataframe(
                [{"run_id": r["run_id"], "target": r["target"],
                  "pass_rate": f"{r['pass_rate']:.0%}", "passed": r["passed"],
                  "partial": r["partial"], "failed": r["failed"],
                  "errors": r["errors"], "total": r["total"],
                  "started_at": r["started_at"]} for r in runs],
                width='stretch', height=360)

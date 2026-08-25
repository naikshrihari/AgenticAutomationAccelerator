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
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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


def ingest_uploads(files, chunk_size: int, settings: Settings):
    """Save uploaded files to a temp folder and turn them into tagged chunks."""
    tmp = Path(tempfile.mkdtemp(prefix="agentprobe_upload_"))
    chunks = []
    for uploaded in files:
        dest = tmp / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        if dest.suffix.lower() not in SUPPORTED_SUFFIXES:
            st.warning(f"Skipped unsupported file: {uploaded.name}")
            continue
        doc = load_document(dest)
        doc.text = clean_text(doc.text)
        chunks.extend(chunk_document(doc, max_tokens=chunk_size))
    return tag_chunks(chunks)


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


def cases_to_records(cases) -> list[dict]:
    """Flatten test cases into rows for a table / CSV."""
    return [
        {
            "question": c.question,
            "expected_answer": c.expected_answer,
            "type": c.question_type.value,
            "difficulty": c.difficulty.value,
            "language": c.language,
            "citation": c.citation,
            "required_facts": " | ".join(c.required_facts),
        }
        for c in cases
    ]


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
    st.text_input("Ollama URL", value="http://localhost:11434/v1", key="ollama_url")
    st.text_input("Generation model", value="llama3.2:3b", key="gen_model")
    st.text_input("Judge model", value="llama3.2:3b", key="judge_model")
    st.text_input("Embedding model", value="nomic-embed-text", key="embed_model")
    st.text_input("Data folder", value="data", key="data_dir")

    if st.button("🔌 Test Ollama connection", use_container_width=True):
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

tab_generate, tab_browse, tab_evaluate = st.tabs(
    ["① Generate questions", "② Browse golden sets", "③ Evaluate an agent"]
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

    if st.button("🚀 Generate", type="primary", disabled=not files):
        settings = build_settings()

        ok, detail = check_ollama(settings)
        if not ok:
            st.error(f"Ollama is not reachable at {settings.ollama_base_url}. "
                     f"Start it and pull your models first.\n\n{detail}")
            st.stop()
        if not types:
            st.error("Pick at least one question type.")
            st.stop()

        with st.status("Working…", expanded=True) as status:
            st.write("📄 Reading and chunking documents…")
            chunks = ingest_uploads(files, chunk_size, settings)
            if limit:
                chunks = chunks[: int(limit)]
            if not chunks:
                status.update(label="No text found", state="error")
                st.error("No readable text was extracted. If your PDF is a scan, it needs OCR first.")
                st.stop()
            st.write(f"Found **{len(chunks)}** sections. "
                     f"Generating up to **{len(chunks) * len(types)}** questions "
                     f"({len(types)} type(s) each)…")

            requirements = extract_requirements(req_file)
            if requirements:
                st.write(f"🎯 Steering questions with business requirements ({len(requirements)} chars).")

            client = OllamaClient(settings, model=settings.generation_model)
            generator = CaseGenerator(
                client=client,
                settings=settings,
                type_mix=[QuestionType(t) for t in types],
                max_workers=int(workers),
                requirements=requirements,
            )

            progress = st.progress(0.0)
            live = st.empty()
            cases = []
            # Drive generation chunk-by-chunk so the UI shows live progress.
            for i, chunk in enumerate(chunks, start=1):
                cases.extend(generator.generate_for_chunk(chunk))
                progress.progress(i / len(chunks))
                live.write(f"Section {i}/{len(chunks)} — {len(cases)} questions so far")

            if do_validate and cases:
                st.write("🔎 Self-consistency check…")
                checker = SelfConsistencyChecker(settings=settings)
                by_id = {c.chunk_id: c for c in chunks}
                cases = checker.filter(cases, by_id)
            if do_dedupe and cases:
                st.write("🧹 De-duplicating…")
                cases = Deduplicator(EmbeddingModel(settings), settings).dedupe(cases)

            cases = ReviewGate().run(cases)  # auto-approve into the golden set
            store = GoldenSetStore(settings.golden_dir)
            saved_path = store.save_version(cases)
            status.update(label=f"Done — {len(cases)} questions generated", state="complete")

        st.session_state["last_cases"] = cases
        st.session_state["last_path"] = str(saved_path)
        st.success(f"Saved golden set to `{saved_path}`")

    # Show the most recent result (persists across reruns)
    if st.session_state.get("last_cases"):
        cases = st.session_state["last_cases"]
        records = cases_to_records(cases)
        st.markdown(f"### Generated questions ({len(records)})")
        st.dataframe(records, use_container_width=True, height=400)

        jsonl = "\n".join(c.model_dump_json() for c in cases).encode("utf-8")
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download CSV", records_to_csv(records),
                           file_name="questions.csv", mime="text/csv", use_container_width=True)
        c2.download_button("⬇️ Download JSONL", jsonl,
                           file_name="golden_set.jsonl", mime="application/json",
                           use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 2 — Browse existing golden sets
# --------------------------------------------------------------------------- #
with tab_browse:
    st.subheader("Browse saved golden sets")
    settings = build_settings()
    store = GoldenSetStore(settings.golden_dir)
    versions = store.versions()
    if not versions:
        st.info(f"No golden sets found under `{settings.golden_dir}`. Generate one first.")
    else:
        choice = st.selectbox("Golden set version", options=list(reversed(versions)))
        if choice:
            cases = store.load(settings.golden_dir / choice)
            records = cases_to_records(cases)
            st.markdown(f"**{len(records)}** questions in `{choice}`")
            st.dataframe(records, use_container_width=True, height=420)
            st.download_button("⬇️ Download CSV", records_to_csv(records),
                               file_name=f"{choice}.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# Tab 3 — Evaluate an agent
# --------------------------------------------------------------------------- #
with tab_evaluate:
    st.subheader("Evaluate an agent against a golden set")
    st.write("Point a target config at your agent, then run the latest golden set against it. "
             "Grading (fact coverage + LLM judge) runs on Ollama.")

    config_files = sorted(str(p) for p in Path("config").glob("*.yaml")) if Path("config").exists() else []
    if not config_files:
        st.info("No target configs found in `config/`. Add a YAML describing your agent's endpoint.")
    else:
        target_path = st.selectbox("Target config", options=config_files)
        settings = build_settings()
        versions = GoldenSetStore(settings.golden_dir).versions()
        golden_choice = st.selectbox("Golden set", options=list(reversed(versions)) or ["(none — generate first)"])

        if st.button("▶️ Run evaluation", type="primary", disabled=not versions):
            from agentprobe.pipeline import Pipeline

            target = TargetConfig.from_yaml(target_path)
            cases = GoldenSetStore(settings.golden_dir).load(settings.golden_dir / golden_choice)
            with st.spinner(f"Running {len(cases)} cases against {target.name}…"):
                try:
                    summary = Pipeline(settings).evaluate(target, cases=cases)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Evaluation failed: {exc}")
                    st.stop()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pass rate", f"{summary.pass_rate:.0%}")
            m2.metric("Passed", summary.passed)
            m3.metric("Failed", summary.failed)
            m4.metric("Infra errors", summary.errors)

            report = settings.reports_dir / summary.run_id / "report.html"
            if report.exists():
                components.html(report.read_text(encoding="utf-8"), height=600, scrolling=True)
                st.download_button("⬇️ Download HTML report", report.read_bytes(),
                                   file_name="report.html", mime="text/html")

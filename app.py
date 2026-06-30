"""
Streamlit frontend for PersonaCast.

The form's only job is to build a validated Persona; everything downstream runs headless through
run_pipeline() (same entry the CLI uses). Script generation and audio are separate steps: Generate
makes the script (~1-3 min, real API calls), and a separate button renders audio with Kokoro TTS so
the slow local synth is opt-in. The finished run is stashed in st.session_state so clicking the
audio button doesn't re-run the whole pipeline.

Run:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from personacast.llm.client import LLMClient
from personacast.models import Expertise, Interest, Persona
from personacast.pipeline import state as state_mod
from personacast.pipeline import tts
from personacast.pipeline.qa import answer_question, flatten_curated
from personacast.pipeline.run import run_pipeline

st.set_page_config(page_title="PersonaCast", layout="wide")
st.title("PersonaCast")
st.caption("Enter a persona, generate a personalized podcast script.")

with st.sidebar:
    st.header("Persona")
    persona_id = st.text_input("Persona id", value="ruhani")

    st.subheader("Interests (per-topic expertise)")
    n = st.number_input("How many topics?", min_value=1, max_value=8, value=3)
    interests: list[Interest] = []
    for i in range(int(n)):
        c1, c2 = st.columns([2, 1])
        topic = c1.text_input(f"Topic {i + 1}", key=f"topic_{i}")
        level = c2.selectbox("Expertise", [e.value for e in Expertise], index=1, key=f"exp_{i}")
        if topic.strip():
            interests.append(Interest(topic=topic.strip(), expertise=Expertise(level)))

    tone = st.text_input("Tone", value="technical but conversational, like talking to a peer")
    avoid_raw = st.text_area("Avoid (one per line)", value="basic ML 101 explanations")
    avoid = [line.strip() for line in avoid_raw.splitlines() if line.strip()]

    st.divider()
    st.caption("Generate makes real API calls — the full script run takes about 6-7 minutes.")
    go = st.button("Generate script", type="primary")


# --- Generate (script only) ---
if go:
    if not interests:
        st.error("Add at least one topic.")
        st.stop()

    persona = Persona(persona_id=persona_id, interests=interests, tone=tone, avoid=avoid)
    try:
        # A simple live ticker: on_stage writes each stage into the status panel as it begins.
        with st.status("Running pipeline…", expanded=True) as status:
            result = run_pipeline(
                persona, audio=False,
                on_stage=lambda label: st.write(f"→ {label}"),
            )
            # Persist so the audio button (which reruns the script) doesn't re-run the pipeline.
            st.session_state["result"] = result
            st.session_state["persona_id"] = persona_id
            st.session_state.pop("audio_path", None)  # new script => any old audio is stale
            st.session_state.pop("qa_answer", None)  # new episode => any old answer is stale
            status.update(label=f"Done — run {result.run_id}", state="complete")
    except Exception as err:  # noqa: BLE001 — surface failures in the UI, don't crash
        st.error(f"Pipeline failed: {type(err).__name__}: {err}")
        st.stop()


# --- Results viewer (reads the stored run; survives reruns from the audio button) ---
result = st.session_state.get("result")
if result is not None:
    st.header("Episode script")
    st.write(result.script or "(no script)")
    if result.script:
        words = len(result.script.split())
        st.caption(f"~{words} words · ~{words / 155:.1f} min at Kokoro's pace")
        st.download_button(
            "Download script.txt", result.script,
            file_name=f"{st.session_state.get('persona_id', 'persona')}_script.txt",
            mime="text/plain",
        )

    # --- Mid-podcast Q&A: ask a question against this episode's curated sources ---
    st.header("Ask a question")
    st.caption(
        "Type a question as if you paused the episode. It's answered from this episode's "
        "curated sources; if they don't cover it (and web fallback is on), it searches the web."
    )
    question = st.text_input("Your question", key="qa_question")
    allow_web = st.checkbox(
        "Search the web if the sources don't cover it", value=True, key="qa_allow_web"
    )
    if st.button("Ask") and question.strip():
        try:
            with st.spinner("Answering…"):
                st.session_state["qa_answer"] = answer_question(
                    question.strip(), result.persona,
                    flatten_curated(result.curated), LLMClient(),
                    allow_web=allow_web,
                )
        except Exception as err:  # noqa: BLE001 — surface failures in the UI
            st.error(f"Q&A failed: {type(err).__name__}: {err}")

    # render the stored answer (survives reruns from other buttons until a new run)
    ans = st.session_state.get("qa_answer")
    if ans is not None:
        if ans.answered:
            st.markdown(ans.answer)
            st.caption(
                "↳ answered from a web search"
                if ans.used_web
                else "↳ answered from this episode's curated sources"
            )
            if ans.sources_used:
                st.markdown("**Sources**")
                for src in ans.sources_used:
                    st.markdown(f"- [{src.title}]({src.url})")
        else:
            st.warning(ans.answer)

    # --- Audio: a separate, explicit step (local TTS is slow) ---
    st.header("Audio")
    st.caption(
        "Optional — renders the script above with Kokoro TTS, about 3 minutes extra on CPU. "
        "✅ This works when you run the app locally (`streamlit run app.py`). "
        "It can't be deployed on Streamlit Community Cloud, though, because the ~350 MB voice model "
        "exceeds GitHub's 100 MB file limit and Kokoro needs the espeak-ng system package, which the "
        "free tier doesn't provide."
    )
    if st.button("Generate audio from this script"):
        try:
            with st.spinner("Rendering audio… (this can take a few minutes)"):
                out = state_mod.run_dir(result.run_id) / "episode.wav"
                audio_path = tts.synthesize(result.script, out)
            st.session_state["audio_path"] = str(audio_path)
        except Exception as err:  # noqa: BLE001
            st.warning(f"Audio skipped — {type(err).__name__}: {err}")
    if st.session_state.get("audio_path"):
        st.audio(st.session_state["audio_path"])

    # --- Per-topic detail: segment + queries + sources that survived curation ---
    st.header("Per-topic detail")
    for seg in result.segments:
        with st.expander(seg.topic, expanded=False):
            st.markdown("**Segment**")
            st.write(seg.text)

            qs = result.queries.get(seg.topic, [])
            if qs:
                st.markdown("**Search queries used**")
                for q in qs:
                    st.markdown(f"- {q}")

            st.markdown("**Sources used**")
            srcs = result.curated.get(seg.topic, [])
            if srcs:
                for src in srcs:
                    st.markdown(f"- [{src.title}]({src.url}) — *{src.source}*")
            else:
                st.caption("(no sources survived curation for this topic)")

    if result.notes:
        st.info("Notes / known gaps: " + "; ".join(result.notes))

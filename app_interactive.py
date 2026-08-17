"""
PersonaCast — Interactive session frontend (v6, agentic retrieval).

Build a persona + this-session context, then generate ~60s turns one at a time. Each turn is
voiced (audix player). You can PAUSE mid-turn and react: the sentence playing at the pause point
becomes the anchor, which is linked back to the source it came from so the NEXT turn goes deeper
on exactly that source. Reaction TYPE is still inferred (text ending in '?' -> question with an
inline grounded answer; empty -> no reaction; otherwise -> comment) and folded into the
persistent per-persona memory, which steers the next turn.

WHAT CHANGED IN v6
------------------
The source pool can now be built by a LangGraph agent instead of the fixed linear loop, and
this frontend exposes that:

  * Retrieval mode toggle in the sidebar. It writes config.AGENTIC_RETRIEVAL at runtime rather
    than requiring an env var, which works because build_source_pool reads the flag at CALL
    time, not at import time. Flipping it to Linear gives you the exact pre-v6 behaviour, so
    you can A/B the two on the same persona in one sitting.
  * A "What the agent decided" panel, rendered from InteractiveSession.retrieval_trace: which
    sources it chose per topic and why, how many it kept, and its full notes trail. Without
    this the agent is a black box, because build_source_pool only returns the pool.
  * The memory panel now shows `covered` — how many sources this listener has already been
    shown per topic. That is the cross-session novelty mechanism: on the NEXT session the
    search node drops those URLs for free, before anything spends an LLM call on them.

Nothing about turn generation, reactions, anchoring, TTS or STT changed.

Run:  streamlit run app_interactive.py
"""

from __future__ import annotations

import re

import streamlit as st
from streamlit_advanced_audio import audix

from personacast import config
from personacast.models import Expertise, Interest, Persona, ReactionType
from personacast.pipeline import interactive as interactive_mod
from personacast.pipeline import state as state_mod
from personacast.pipeline import stt
from personacast.pipeline import tts
from personacast.pipeline.interactive import InteractiveSession


def _split_sentences(text: str) -> list[str]:
    """Split a turn into the sentences the listener can pin a reaction to."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


st.set_page_config(page_title="PersonaCast — Interactive", layout="wide")
st.title("PersonaCast — Interactive")
st.caption("Generate ~60s turns, react, and watch the persona's memory adapt.")


# --------------------------------------------------------------------------- sidebar: persona
with st.sidebar:
    st.header("Persona")
    persona_id = st.text_input("Name (memory is keyed to this)", value="ruhani1")

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
    avoid_raw = st.text_area("Avoid (one per line)", value="")
    avoid = [line.strip() for line in avoid_raw.splitlines() if line.strip()]
    context = st.text_input("This session's context / vibe", value="walking in the park, relaxed")

    # ----------------------------------------------------------------- retrieval mode (new)
    st.divider()
    st.subheader("Retrieval")
    mode = st.radio(
        "How to build the source pool",
        ["Agentic (LangGraph)", "Linear (original)"],
        index=0,
        help=(
            "Agentic: a model picks web vs arXiv per topic and writes the queries, primed with "
            "what you've already been told, and skips links you were shown in a past session.\n\n"
            "Linear: the original fixed loop. Sources are chosen by a keyword scan over the "
            "topic string."
        ),
    )
    agentic = mode.startswith("Agentic")

    if agentic:
        st.caption(
            "Same number of LLM calls as Linear — the model just picks the sources instead of "
            "a keyword scan, and skips links you were shown in a past session."
        )

    st.subheader("Interaction")
    agentic_interaction = st.checkbox(
        "Read reactions semantically", value=config.AGENTIC_INTERACTION,
        help=(
            "On: one LLM call reads each reaction for intent, sentiment, a continuous "
            "engagement score, and topic-switch requests by MEANING. Costs no extra time — the "
            "same call also returns the turn gist, replacing a call already being made.\n\n"
            "Off: reaction type is decided by whether the text ends in '?', topic switches by a "
            "whole-word regex, and engagement by four fixed constants."
        ),
    )

    st.divider()
    speak = st.checkbox("🔊 Speak each turn (local TTS)", value=True)
    st.caption("Start builds the source pool once, with real API calls. ~1-3 min.")
    start = st.button("Start session", type="primary")


# --------------------------------------------------------------------------- start a session
if start:
    if not interests:
        st.error("Add at least one topic.")
        st.stop()

    # set the flag BEFORE start(), since build_source_pool reads config at call time.
    # this is what lets the sidebar toggle work without restarting streamlit.
    config.AGENTIC_RETRIEVAL = agentic
    config.AGENTIC_INTERACTION = agentic_interaction

    persona = Persona(
        persona_id=persona_id, interests=interests, tone=tone, avoid=avoid,
        additional_context=context,
    )
    try:
        session = InteractiveSession(persona, on_stage=lambda label: st.write(f"→ {label}"))
        with st.status("Building the source pool…", expanded=True) as status:
            session.start()
            label = f"Pool ready — run {session.state.run_id}"
            if session.retrieval_trace:
                kept = sum(t["kept"] for t in session.retrieval_trace.values())
                label += f" · {kept} sources across {len(session.retrieval_trace)} topics"
            status.update(label=label, state="complete")
        st.session_state["session"] = session
        st.session_state["agentic"] = agentic
        st.session_state["current_turn"] = session.next_segment()
        st.session_state.pop("last_answer", None)
        st.session_state.pop("audio_path", None)
    except Exception as err:  # noqa: BLE001 — surface failures in the UI
        st.error(f"Failed to start: {type(err).__name__}: {err}")
        st.stop()


session: InteractiveSession | None = st.session_state.get("session")


# a cute character that "talks" (bobs/pulses) while a turn is playing
_SPEAKER_HTML = """
<style>
@keyframes pc-talk {
  0%,100% { transform: translateY(0) scale(1); }
  25%     { transform: translateY(-7px) scale(1.06); }
  50%     { transform: translateY(0) scale(0.97); }
  75%     { transform: translateY(-4px) scale(1.03); }
}
.pc-wrap { text-align:center; padding:8px 0; }
.pc-speaker { font-size:84px; line-height:1; display:inline-block; animation: pc-talk .6s ease-in-out infinite; }
.pc-cap { color:#8a8a8a; font-size:13px; margin-top:2px; }
</style>
<div class="pc-wrap"><span class="pc-speaker">🐧</span><div class="pc-cap">🔊 speaking…</div></div>
"""


def _play_turn(sess: InteractiveSession, turn) -> None:
    """
    Voice this turn with the local piper TTS, show a talking character, and autoplay the audio via
    the audix player. Synthesized audio is cached per iteration so incidental reruns don't
    re-synth, and autoplay only fires on a NEW turn (guarded by last_played_iter) so reruns
    don't replay. audix reports the real playback currentTime on pause/scrub — we stash the
    latest so a mid-turn pause can anchor the reaction to the sentence being spoken.
    """
    cache = st.session_state.setdefault("turn_audio", {})
    if turn.iteration not in cache:
        try:
            with st.spinner("Voicing this turn…"):
                out = state_mod.run_dir(sess.state.run_id) / f"turn_{turn.iteration}.wav"
                cache[turn.iteration] = str(tts.synthesize(turn.text, out))
        except Exception as err:  # noqa: BLE001 — audio is best-effort; never block the turn
            st.warning(f"TTS failed: {type(err).__name__}: {err}")
            cache[turn.iteration] = None
    path = cache.get(turn.iteration)
    if not path:
        return
    st.markdown(_SPEAKER_HTML, unsafe_allow_html=True)
    autoplay = st.session_state.get("last_played_iter") != turn.iteration
    result = audix(path, key=f"audix_{turn.iteration}", autoplay=autoplay)
    st.session_state["last_played_iter"] = turn.iteration
    if result and result.get("currentTime") is not None:
        st.session_state[f"pos_{turn.iteration}"] = result.get("currentTime")


def _render_retrieval(sess: InteractiveSession) -> None:
    """
    What the retrieval agent decided, per topic.

    Everything here comes from InteractiveSession.retrieval_trace, which is filled by the
    on_topic_done hook during start(). It is empty on the linear path, because the linear path
    makes no decisions worth reporting — the source choice there is a substring scan.

    The number to watch is `kept` — a topic with 2 or 3 sources will run dry after a couple of
    turns, and this is the only place that is visible before it happens.

    Deliberately NOT shown here: a comparison against retrieve._wants_arxiv. Whether the agent
    disagrees with the keyword scan is a one-off evaluation question, not a per-render one, and
    it lives in the smoke test (`python -m personacast.agents.graph`). Re-running the scan in
    the UI would imply it is still a reference answer, which is exactly what it is not.
    """
    trace = sess.retrieval_trace
    if not trace:
        if st.session_state.get("agentic"):
            st.caption("No retrieval trace — the agent fell back to the linear path.")
        else:
            st.caption("Linear retrieval — sources chosen by keyword scan, no agent trace.")
        return

    st.subheader("Retrieval — what the agent decided")
    for topic, info in trace.items():
        srcs = " + ".join(info["sources"]) or "?"
        thin = "⚠️ " if info["kept"] < 3 else ""

        with st.expander(f"{thin}{topic} · {srcs} · {info['kept']} sources", expanded=False):
            st.caption(f"{info['kept']} kept of {info['retrieved']} retrieved")

            st.markdown("**Queries searched**")
            for q in info["queries"]:
                st.markdown(f"- `{q}`")
            if info.get("arxiv_queries"):
                st.markdown("**arXiv variants** (academic phrasing)")
                for q in info["arxiv_queries"]:
                    st.markdown(f"- `{q}`")

            st.markdown("**Trace**")
            for note in info["notes"]:
                st.markdown(f"- {note}")


def _render_memory(sess: InteractiveSession) -> None:
    """live engagement panel + what's already been covered + reaction log."""
    mem = sess.state.memory
    active = [i.topic for i in sess.persona.interests]
    scores = {t: mem.engagement.get(t, 0.0) for t in active}

    st.subheader("Memory — engagement")
    peak = max(scores.values(), default=1.0) or 1.0
    for topic, pts in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        st.caption(f"{topic} · {pts:g} pts")
        st.progress(min(pts / peak, 1.0))

    # cross-session source history. this is what the retrieval agent reads NEXT session to
    # avoid re-serving the same links, so it is worth being able to see it accumulate.
    covered = {t: mem.covered.get(t, []) for t in active if mem.covered.get(t)}
    if covered:
        total = sum(len(v) for v in covered.values())
        with st.expander(f"Already covered ({total} sources)", expanded=False):
            st.caption("Dropped for free by the search step in your next session.")
            for topic, sources in covered.items():
                st.markdown(f"**{topic}** — {len(sources)}")
                for source in sources[-6:]:
                    st.markdown(f"- {source.title}")

    if mem.reactions:
        with st.expander(f"Reaction history ({len(mem.reactions)})", expanded=False):
            # when the interaction agent is on, show what it READ, not just the enum it mapped
            # to. the spread of engagement_delta is the thing to watch: if every value is
            # +2/+1/-1 the agent is just reproducing the heuristic table it replaced.
            for r in mem.reactions[-12:]:
                icon = {ReactionType.question: "❓", ReactionType.comment: "💬", ReactionType.none: "·"}[r.type]
                pin = f" · 📍→ {r.anchor_source}" if r.anchor_source else (" · 📍" if r.anchor_snippet else "")
                st.markdown(f"{icon} **{r.topic}**{pin} — {r.text or '(no reaction)'}")
                if r.intent:
                    bits = [f"`{r.intent}`"]
                    if r.sentiment is not None:
                        bits.append(f"sentiment {r.sentiment:+.1f}")
                    if r.engagement_delta is not None:
                        bits.append(f"**{r.engagement_delta:+.1f} pts**")
                    st.caption(" · ".join(bits))


# --------------------------------------------------------------------------- main: turn + reaction
if session is not None:
    left, right = st.columns([2, 1])

    with right:
        _render_retrieval(session)
        st.divider()
        _render_memory(session)

    with left:
        heard = st.session_state.get("heard")
        if heard:
            st.caption(f"🎤 heard: \"{heard}\"")

        last_answer = st.session_state.get("last_answer")
        if last_answer:
            st.caption("↳ Your question is answered inside this turn.")
            with st.expander("grounded answer used (provenance)", expanded=False):
                st.write(last_answer)

        turn = st.session_state.get("current_turn")
        if turn is not None:
            words = len(turn.text.split())
            st.subheader(f"Turn {turn.iteration} · {turn.topic}")

            # how much material this turn actually had behind it. a thin pool means this topic
            # will run dry after a couple of turns, so surface it rather than letting it hide.
            n_sources = len(session.state.pool.get(turn.topic, []))
            st.caption(f"~{words} words · ~{words / 155 * 60:.0f}s · {n_sources} sources in pool")
            st.write(turn.text)

            if speak:
                _play_turn(session, turn)

            # --- pause-driven anchor: the sentence playing when the listener paused ----------
            # audix reports the real pause currentTime; map it (linearly) to a sentence. Any
            # pause auto-selects that sentence (the anchor only actually drives a deep-dive if a
            # reaction is typed/spoken — a bare pause is dropped in submit_reaction).
            path = st.session_state.get("turn_audio", {}).get(turn.iteration)
            pos = st.session_state.get(f"pos_{turn.iteration}")
            dur = tts.wav_duration(path) if path else 0.0
            sentences = _split_sentences(turn.text)
            _whole = "(react to the whole turn)"
            options = [_whole] + sentences
            akey = f"anchor_{turn.iteration}"
            # a NEW pause writes the detected sentence straight into the widget's state — we can't
            # use index= because Streamlit ignores it once a keyed widget exists. Guarded on the
            # position so a manual override afterwards sticks until the listener pauses again.
            if pos and dur:
                detected = interactive_mod.locate_snippet(turn.text, pos, dur)
                if detected in sentences and st.session_state.get(f"anchored_pos_{turn.iteration}") != pos:
                    st.session_state[akey] = detected
                    st.session_state[f"anchored_pos_{turn.iteration}"] = pos
            anchor_choice = st.selectbox(
                "⏸ You interrupted around here — adjust if needed:",
                options,
                key=akey,
            )
            anchor_snippet = "" if anchor_choice == _whole else anchor_choice
            if pos:
                st.caption(f"paused at ~{pos:.0f}s")

            react = st.text_input(
                "Type a reaction (end with '?' to ask · leave empty for no reaction)",
                key=f"react_{turn.iteration}",
            )
            audio = st.audio_input("🎤 …or speak your reaction", key=f"mic_{turn.iteration}")
            c1, c2 = st.columns([1, 4])
            if c1.button("React & continue", key=f"go_{turn.iteration}"):
                try:
                    # voice wins if recorded: transcribe it (local whisper STT) into the reaction text,
                    # which then flows through the same classify/switch pipeline as typed text
                    reaction_text = react
                    if audio is not None:
                        with st.spinner("Transcribing your voice…"):
                            reaction_text = stt.transcribe(audio.getvalue())
                        st.session_state["heard"] = reaction_text
                    # pass the pinned sentence (if any) so the reaction is anchored to that exact
                    # point, driving the next turn's deep-dive
                    done_turn = session.submit_reaction(reaction_text, anchor_snippet=anchor_snippet)
                    st.session_state["last_answer"] = (
                        done_turn.reaction.answer
                        if done_turn.reaction and done_turn.reaction.type == ReactionType.question
                        else None
                    )
                    if session.done:
                        session.finish()
                        st.session_state["current_turn"] = None
                    else:
                        st.session_state["current_turn"] = session.next_segment()
                    st.rerun()
                except Exception as err:  # noqa: BLE001
                    st.error(f"Turn failed: {type(err).__name__}: {err}")
            if c2.button("End session", key=f"end_{turn.iteration}"):
                session.finish()
                st.session_state["current_turn"] = None
                st.rerun()
        else:
            st.success(f"Session complete — {len(session.state.turns)} turns.")
            st.caption(
                "Your `covered` history was just saved. Start another session with the same "
                "persona name and the retrieval agent will skip these sources."
            )
            transcript = "\n\n".join(f"[{t.topic}] {t.text}" for t in session.state.turns)
            st.download_button("Download transcript", transcript,
                               file_name=f"{session.persona.persona_id}_session.txt")

            # optional audio of the whole session (local piper TTS)
            if st.button("Generate audio of this session"):
                try:
                    with st.spinner("Synthesizing…"):
                        out = state_mod.run_dir(session.state.run_id) / "episode.wav"
                        st.session_state["audio_path"] = str(tts.synthesize(transcript, out))
                except Exception as err:  # noqa: BLE001
                    st.warning(f"Audio failed: {type(err).__name__}: {err}")
            if st.session_state.get("audio_path"):
                st.audio(st.session_state["audio_path"])
else:
    st.info("Build a persona in the sidebar and click **Start session**.")

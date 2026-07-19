"""
PersonaCast — Interactive session frontend (v4).

Build a persona + this-session context, then generate ~60s turns one at a time. After each
turn you type a reaction; its TYPE is inferred (text ending in '?' -> question and gets an
inline grounded answer; empty -> no reaction; otherwise -> comment) and folded into the
persistent per-persona memory, which steers the next turn. A live memory panel shows the
engagement points growing.

Run:  streamlit run app_interactive.py
"""

from __future__ import annotations

import streamlit as st

from personacast import config
from personacast.models import Expertise, Interest, Persona, ReactionType
from personacast.pipeline import state as state_mod
from personacast.pipeline import stt
from personacast.pipeline import tts
from personacast.pipeline.interactive import InteractiveSession

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

    st.divider()
    speak = st.checkbox("🔊 Speak each turn (Live TTS)", value=True)
    st.caption("Start builds the source pool once (real API calls, ~1-3 min).")
    start = st.button("Start session", type="primary")


# --------------------------------------------------------------------------- start a session
if start:
    if not interests:
        st.error("Add at least one topic.")
        st.stop()
    persona = Persona(
        persona_id=persona_id, interests=interests, tone=tone, avoid=avoid,
        additional_context=context,
    )
    try:
        session = InteractiveSession(persona, on_stage=lambda label: st.write(f"→ {label}"))
        with st.status("Building the source pool…", expanded=True) as status:
            session.start()
            status.update(label=f"Pool ready — run {session.state.run_id}", state="complete")
        st.session_state["session"] = session
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
    Voice this turn with the Live-API TTS, show a talking character, and autoplay the audio.
    Synthesized audio is cached per iteration so incidental reruns don't re-synth, and
    autoplay only fires on a NEW turn (guarded by last_played_iter) so reruns don't replay.
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
    st.audio(path, autoplay=autoplay)
    st.session_state["last_played_iter"] = turn.iteration


def _render_memory(sess: InteractiveSession) -> None:
    """live engagement panel + reaction log."""
    mem = sess.state.memory
    active = [i.topic for i in sess.persona.interests]
    scores = {t: mem.engagement.get(t, 0.0) for t in active}
    st.subheader("Memory — engagement")
    peak = max(scores.values(), default=1.0) or 1.0
    for topic, pts in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        st.caption(f"{topic} · {pts:g} pts")
        st.progress(min(pts / peak, 1.0))
    if mem.reactions:
        with st.expander(f"Reaction history ({len(mem.reactions)})", expanded=False):
            for r in mem.reactions[-12:]:
                icon = {ReactionType.question: "❓", ReactionType.comment: "💬", ReactionType.none: "·"}[r.type]
                st.markdown(f"{icon} **{r.topic}** — {r.text or '(no reaction)'}")


# --------------------------------------------------------------------------- main: turn + reaction
if session is not None:
    left, right = st.columns([2, 1])

    with right:
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
            st.caption(f"~{words} words · ~{words / 155 * 60:.0f}s")
            st.write(turn.text)

            if speak:
                _play_turn(session, turn)

            react = st.text_input(
                "Type a reaction (end with '?' to ask · leave empty for no reaction)",
                key=f"react_{turn.iteration}",
            )
            audio = st.audio_input("🎤 …or speak your reaction", key=f"mic_{turn.iteration}")
            c1, c2 = st.columns([1, 4])
            if c1.button("React & continue", key=f"go_{turn.iteration}"):
                try:
                    # voice wins if recorded: transcribe it (Live-API STT) into the reaction text,
                    # which then flows through the same classify/switch pipeline as typed text
                    reaction_text = react
                    if audio is not None:
                        with st.spinner("Transcribing your voice…"):
                            reaction_text = stt.transcribe(audio.getvalue())
                        st.session_state["heard"] = reaction_text
                    done_turn = session.submit_reaction(reaction_text)
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
            transcript = "\n\n".join(f"[{t.topic}] {t.text}" for t in session.state.turns)
            st.download_button("Download transcript", transcript,
                               file_name=f"{session.persona.persona_id}_session.txt")

            # optional audio of the whole session (Gemini Live-API TTS)
            if st.button("Generate audio of this session"):
                try:
                    with st.spinner("Synthesizing with Gemini Live TTS…"):
                        out = state_mod.run_dir(session.state.run_id) / "episode.wav"
                        st.session_state["audio_path"] = str(tts.synthesize(transcript, out))
                except Exception as err:  # noqa: BLE001
                    st.warning(f"Audio failed: {type(err).__name__}: {err}")
            if st.session_state.get("audio_path"):
                st.audio(st.session_state["audio_path"])
else:
    st.info("Build a persona in the sidebar and click **Start session**.")

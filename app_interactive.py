
from __future__ import annotations

import re
import time

import streamlit as st
from streamlit_advanced_audio import audix

from personacast import config
from personacast.models import Expertise, Interest, Persona, ReactionType
from personacast.pipeline import interactive as interactive_mod
from personacast.pipeline import state as state_mod
from personacast.pipeline import stt
from personacast.pipeline import timing
from personacast.pipeline import tts
from personacast.pipeline.interactive import InteractiveSession


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


st.set_page_config(page_title="PersonaCast — Interactive", layout="wide")
st.title("PersonaCast — Interactive")
st.caption("Generate ~60s turns, react, and watch the persona's memory adapt.")


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
    st.subheader("Source pool")
    rebuild = st.checkbox(
        "Rebuild from scratch", value=False,
        help=(
            "The pool is cached to disk per persona and reused for "
            f"{config.POOL_CACHE_TTL_HOURS:.0f}h. It is the biggest consumer of the daily "
            "request cap and takes 1-3 minutes, and none of that work is about the "
            "interruption path — so by default we skip it and spend the budget on the session. "
            "Tick this when you actually want fresh material."
        ),
    )

    st.divider()
    speak = st.checkbox("🔊 Speak each turn (local TTS)", value=True)
    st.caption(
        "Cached pool: start is instant. Rebuilding: ~1-3 min of real API calls."
    )
    start = st.button("Start session", type="primary")


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
            session.start(rebuild_pool=rebuild)
            label = f"Pool ready — run {session.state.run_id}"
            if session.pool_from_cache:
                total = sum(len(v) for v in session.state.pool.values())
                label += f" · {total} sources from cache, 0 retrieval calls"
            if session.retrieval_trace:
                kept = sum(t["kept"] for t in session.retrieval_trace.values())
                label += f" · {kept} sources across {len(session.retrieval_trace)} topics"
            status.update(label=label, state="complete")
        st.session_state["session"] = session
        st.session_state["current_turn"] = session.next_segment()
        st.session_state.pop("last_answer", None)
        st.session_state.pop("audio_path", None)
    except Exception as err:
        st.error(f"Failed to start: {type(err).__name__}: {err}")
        st.stop()


session: InteractiveSession | None = st.session_state.get("session")


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


def _render_retrieval(sess: InteractiveSession) -> None:
    trace = sess.retrieval_trace
    if not trace:
        st.caption("No retrieval trace — the planning agent errored out and fell back to keywords.")
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
    mem = sess.state.memory
    active = [i.topic for i in sess.persona.interests]
    scores = {t: mem.engagement.get(t, 0.0) for t in active}

    st.subheader("Memory — engagement")
    peak = max(scores.values(), default=1.0) or 1.0
    for topic, pts in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        st.caption(f"{topic} · {pts:g} pts")
        st.progress(min(pts / peak, 1.0))

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


def _bridge_remaining() -> float:
    started = st.session_state.get("bridge_started_at")
    if started is None:
        return 0.0
    return max(0.0, st.session_state.get("bridge_duration", 0.0) - (time.monotonic() - started))


def _play(path: str | None, key: str, label: str, *, track_pos: str | None = None) -> None:
    if not path:
        return
    st.caption(label)
    fresh = st.session_state.get("last_autoplayed") != key
    result = audix(path, key=f"aud_{key}", autoplay=fresh)
    if fresh:
        st.session_state["last_autoplayed"] = key
    if track_pos and result and result.get("currentTime") is not None:
        st.session_state[track_pos] = result["currentTime"]


@st.fragment(run_every=1.0)
def _playback_tick(sess: InteractiveSession) -> None:
    future = st.session_state.get("pending_future")
    if future is None:
        return

    if not future.done():
        st.caption("✍️ writing and voicing the response…")
        return

    remaining = _bridge_remaining()
    if remaining > 0.05:
        st.caption(f"✅ response ready — playing after the bridge, {remaining:.0f}s")
        return

    try:
        nxt = future.result()
    except Exception as err:
        st.session_state["pending_future"] = None
        st.error(f"Turn failed: {type(err).__name__}: {err}")
        st.rerun(scope="app")
        return

    parts = [c["path"] for c in sess.audio_chunks if c["path"]]
    st.session_state["response_audio"] = parts[0] if parts else None
    st.session_state["pending_future"] = None
    st.session_state["bridge_started_at"] = None
    st.session_state["current_turn"] = nxt
    reaction = sess.state.turns[-2].reaction if nxt and len(sess.state.turns) > 1 else None
    st.session_state["last_answer"] = (
        reaction.answer if reaction and reaction.type == ReactionType.question else None
    )
    st.rerun(scope="app")


def _start_reaction(sess: InteractiveSession, reaction_text: str, anchor_snippet: str) -> None:
    st.session_state.pop("response_audio", None)
    st.session_state.pop("turn_pos", None)

    opener, future = sess.begin_reaction(reaction_text, anchor_snippet=anchor_snippet)

    st.session_state["bridge_audio"] = opener["audio"]
    st.session_state["bridge_started_at"] = time.monotonic()
    st.session_state["bridge_duration"] = (
        tts.wav_duration(opener["audio"]) if opener["audio"] else 0.0
    )
    st.session_state["pending_future"] = future
    st.session_state["opener"] = opener


def _start_reaction_baseline(sess: InteractiveSession, reaction_text: str, anchor_snippet: str) -> None:
    reacted_to = sess.state.turns[-1].iteration
    timer = timing.Timer()
    done_turn = sess.submit_reaction(reaction_text, anchor_snippet=anchor_snippet, timer=timer)
    st.session_state["last_answer"] = (
        done_turn.reaction.answer
        if done_turn.reaction and done_turn.reaction.type == ReactionType.question
        else None
    )
    st.session_state.pop("response_audio", None)
    if sess.done:
        sess.finish()
        st.session_state["current_turn"] = None
        return

    with timer.stage(timing.GENERATE):
        nxt = sess.next_segment()
    try:
        with timer.stage(timing.TTS):
            out = state_mod.run_dir(sess.state.run_id) / f"turn_{nxt.iteration}.wav"
            path = str(tts.synthesize(nxt.text, out))
        st.session_state["response_audio"] = path
        timer.mark(timing.FIRST_AUDIO)
    except Exception as err:
        st.warning(f"TTS failed: {type(err).__name__}: {err}")

    sess.timings[reacted_to] = timer.as_dict()
    st.session_state["current_turn"] = nxt


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

        if session.timings:
            last_iter = max(session.timings)
            marks = session.timings[last_iter].get("marks", {})
            stages = session.timings[last_iter].get("stages", {})
            first = marks.get("time_to_first_audio")
            if first is not None:
                mode = "fast" if config.FAST_INTERACTION else "baseline"
                detail = " · ".join(f"{k} {v:.1f}s" for k, v in
                                    sorted(stages.items(), key=lambda kv: -kv[1]) if v >= 0.05)
                work = marks.get("continuation_ready")
                head = f"⏱ **{first:.2f}s to first audio**"
                if work:
                    head += f" · continuation ready at {work:.1f}s"
                st.success(f"{head} — {detail}")

                if marks.get("llm_calls"):
                    bits = [f"{marks['llm_calls']:.0f} HTTP calls",
                            f"api {marks.get('llm_api_time', 0):.1f}s",
                            f"our throttle {marks.get('llm_rate_wait', 0):.1f}s"]
                    if marks.get("llm_json_retries"):
                        bits.append(f"⚠️ {marks['llm_json_retries']:.0f} JSON retries")
                    if marks.get("llm_http_retries"):
                        bits.append(f"⚠️ {marks['llm_http_retries']:.0f} HTTP retries")
                    st.caption("interpret breakdown: " + " · ".join(bits))

        turn = st.session_state.get("current_turn")
        waiting = st.session_state.get("pending_future") is not None

        if waiting:
            opener = st.session_state.get("opener", {})
            st.subheader("🔊 Bridging…")
            st.caption(
                f"Selected the **{opener.get('class', '?')}** candidate with a string match — "
                "no LLM call ran before this audio started."
            )
            st.write(opener.get("text", ""))
            _play(st.session_state.get("bridge_audio"), f"bridge_{turn.iteration}",
                  f"🌉 bridge · picked *{opener.get('class', '?')}* by string match, no LLM call")
            _playback_tick(session)

        elif turn is not None:
            words = len(turn.text.split())
            st.subheader(f"Turn {turn.iteration} · {turn.topic}")

            n_sources = len(session.state.pool.get(turn.topic, []))
            st.caption(f"~{words} words · ~{words / 155 * 60:.0f}s · {n_sources} sources in pool")
            st.write(turn.text)

            if speak:
                if not st.session_state.get("response_audio"):
                    try:
                        with st.spinner("Voicing this turn…"):
                            out = state_mod.run_dir(session.state.run_id) / f"turn_{turn.iteration}.wav"
                            st.session_state["response_audio"] = str(tts.synthesize(turn.text, out))
                    except Exception as err:
                        st.warning(f"TTS failed: {type(err).__name__}: {err}")
                st.markdown(_SPEAKER_HTML, unsafe_allow_html=True)
                _play(st.session_state.get("response_audio"), f"resp_{turn.iteration}",
                      "🎙 the full response — pause anywhere to anchor your reaction",
                      track_pos=f"pos_{turn.iteration}")

            pool = session.openers.get(turn.iteration)
            with st.expander(
                f"🎲 Candidate openers ready ({len(pool) if pool else 0}/4)"
                + ("" if pool else " — still generating, static fallbacks are armed"),
                expanded=False,
            ):
                for name, text in (pool or session.openers.get(0) or {}).items():
                    st.markdown(f"**{name}** — {text}")
                if not pool:
                    st.caption(
                        "Generated in the background during playback. If the rate-limit window "
                        "is tight this is shed on purpose and the static set is used instead."
                    )

            path = st.session_state.get("response_audio")
            pos = st.session_state.get(f"pos_{turn.iteration}")
            dur = tts.wav_duration(path) if path else 0.0
            sentences = _split_sentences(turn.text)
            _whole = "(react to the whole turn)"
            options = [_whole] + sentences
            akey = f"anchor_{turn.iteration}"
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
                    reaction_text = react
                    if audio is not None:
                        with st.spinner("Transcribing your voice…"):
                            reaction_text = stt.transcribe(audio.getvalue())
                        st.session_state["heard"] = reaction_text

                    if config.FAST_INTERACTION:
                        _start_reaction(session, reaction_text, anchor_snippet)
                    else:
                        with st.spinner("Reading your reaction and writing the next turn…"):
                            _start_reaction_baseline(session, reaction_text, anchor_snippet)
                    st.rerun()
                except Exception as err:
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

            if st.button("Generate audio of this session"):
                try:
                    with st.spinner("Synthesizing…"):
                        out = state_mod.run_dir(session.state.run_id) / "episode.wav"
                        st.session_state["audio_path"] = str(tts.synthesize(transcript, out))
                except Exception as err:
                    st.warning(f"Audio failed: {type(err).__name__}: {err}")
            if st.session_state.get("audio_path"):
                st.audio(st.session_state["audio_path"])
else:
    st.info("Build a persona in the sidebar and click **Start session**.")

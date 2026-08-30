
from __future__ import annotations

import queue
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

    st.divider()
    mic_always_on = st.checkbox(
        "🎙️ Always-on mic (auto pause/resume, no button)", value=config.MIC_ALWAYS_ON,
        help="Requires headphones — there's no echo cancellation, so the mic will pick up the podcast itself through open speakers.",
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
        st.session_state["mic_always_on"] = mic_always_on
        st.session_state["current_turn"] = session.next_segment()
        ### plain arm_mic() rather than _arm_mic_session(): this block runs at module level
        ### before that helper is defined, and there is nothing to flush yet anyway — the
        ### MicSession isn't created until the first _render_always_on_mic call.
        session.arm_mic()
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


def _play(path: str | None, key: str, label: str, *, track_pos: str | None = None,
          playback_override: tuple[float, bool] | None = None) -> dict | None:
    """
    playback_override: optional (start_time, autoplay) forcing a fresh remount at a
    specific position. audix (streamlit_advanced_audio) has NO imperative pause/play
    control -- only mount-time props (autoplay, start_time) -- so there is no way to tell
    an already-playing widget to stop. The only way to change its playback state after
    first mount is to remount it under a NEW key with different props: autoplay=False
    freezes it at start_time (our "pause"), autoplay=True at a later start_time resumes it.

    Returns the raw component result dict (or None) so callers needing more than
    track_pos's currentTime snapshot -- e.g. _render_turn_audio's isPlaying-driven
    stopwatch, see its docstring for why currentTime alone can't be trusted -- can inspect it.

    NORMAL (non-override) PATH always passes autoplay=True for the active key, never
    switching to False after the first render. Confirmed via direct headless-browser
    testing that this matters a lot: autoplay=True on the first render followed by
    autoplay=False on the very next re-render of the SAME key (which is what happens if
    this call is repeated every 0.25s from inside a polling fragment, as _render_turn_audio
    now is) reliably prevented the audio from ever starting at all -- confirmed reproducibly
    with a fake tone file, paused=True forever. Repeating autoplay=True on every re-render
    of an already-playing widget is safe (confirmed: currentTime progresses continuously,
    no restarts) -- it's the True-then-False flip specifically that breaks it, and Chrome's
    autoplay policy is per-page-lifetime (any earlier real click, e.g. "Start session",
    unlocks it), so re-asserting True is exactly what we want.
    """
    if not path:
        return None
    st.caption(label)
    if playback_override is not None:
        start_time, autoplay = playback_override
        widget_key = f"{key}_at_{start_time:.2f}_{autoplay}"
        result = audix(path, key=f"aud_{widget_key}", autoplay=autoplay, start_time=start_time)
        st.session_state["last_autoplayed"] = widget_key
    else:
        result = audix(path, key=f"aud_{key}", autoplay=True)
        st.session_state["last_autoplayed"] = key
    if track_pos and result and result.get("currentTime") is not None:
        st.session_state[track_pos] = result["currentTime"]
    return result


def _render_turn_audio(sess: InteractiveSession, turn) -> None:
    """
    Voices the turn (if not already synthesized) and renders its player, tracking
    pos_{turn.iteration} as an estimated playback position.

    IMPORTANT #1: when the always-on mic is active this must be called from WITHIN
    _mic_tick's @st.fragment(run_every=0.25), not from the outer script body. The outer
    body only re-executes on full-script reruns, which the mic flow only triggers
    sparsely (onset detected / utterance resolved) -- while the user is just listening,
    nothing else re-invokes this, so without the fragment we'd never notice playback
    actually starting/stopping in time.

    IMPORTANT #2: audix's reported `currentTime` does NOT update continuously during
    playback, AND `isPlaying` also proved unreliable in the full app despite working in
    isolated component tests (confirmed: pos stuck at exactly 0.0 for a full 56-second
    turn, meaning isPlaying never registered True there even though audio was audibly
    playing) -- so this component's own state reporting cannot be trusted for tracking
    position AT ALL. Position is instead tracked as a pure wall-clock stopwatch driven
    ENTIRELY by state we ourselves set (mic_playback_override), with zero dependency on
    anything the component reports back.
    """
    if not st.session_state.get("response_audio"):
        try:
            with st.spinner("Voicing this turn…"):
                out = state_mod.run_dir(sess.state.run_id) / f"turn_{turn.iteration}.wav"
                st.session_state["response_audio"] = str(tts.synthesize(turn.text, out))
        except Exception as err:
            st.warning(f"TTS failed: {type(err).__name__}: {err}")
    mic_override = st.session_state.get(f"mic_playback_override_{turn.iteration}")
    is_paused = mic_override is not None and not mic_override[1]
    if is_paused:
        st.caption("⏸ paused — listening…")
    else:
        st.markdown(_SPEAKER_HTML, unsafe_allow_html=True)
    _play(st.session_state.get("response_audio"), f"resp_{turn.iteration}",
          "🎙 the full response — pause anywhere to anchor your reaction",
          playback_override=mic_override)

    ### wall-clock stopwatch, armed/disarmed purely by is_paused (state we set ourselves
    ### in _mic_tick's onset handler / empty-transcript resume -- never by anything the
    ### audix component reports back).
    start_key = f"turn_audio_started_at_{turn.iteration}"
    paused_total_key = f"turn_audio_paused_total_{turn.iteration}"
    pause_began_key = f"turn_audio_pause_began_at_{turn.iteration}"
    was_paused_key = f"turn_audio_was_paused_{turn.iteration}"

    if start_key not in st.session_state:
        st.session_state[start_key] = time.monotonic()
        st.session_state[paused_total_key] = 0.0

    was_paused = st.session_state.get(was_paused_key, False)
    if is_paused and not was_paused:
        st.session_state[pause_began_key] = time.monotonic()
    elif not is_paused and was_paused:
        pause_began = st.session_state.get(pause_began_key)
        if pause_began is not None:
            st.session_state[paused_total_key] = st.session_state.get(paused_total_key, 0.0) + (time.monotonic() - pause_began)
    st.session_state[was_paused_key] = is_paused

    paused_total = st.session_state.get(paused_total_key, 0.0)
    if is_paused:
        pause_began = st.session_state.get(pause_began_key, time.monotonic())
        elapsed = (pause_began - st.session_state[start_key]) - paused_total
    else:
        elapsed = (time.monotonic() - st.session_state[start_key]) - paused_total
    st.session_state[f"pos_{turn.iteration}"] = max(0.0, elapsed)


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
    _arm_mic_session(sess)
    reaction = sess.state.turns[-2].reaction if nxt and len(sess.state.turns) > 1 else None
    st.session_state["last_answer"] = (
        reaction.answer if reaction and reaction.type == ReactionType.question else None
    )
    st.rerun(scope="app")


def _disarm_mic_session(sess: InteractiveSession) -> None:
    """
    §1.6's barge-in guard: stop capturing the INSTANT a reaction is submitted.

    This can't be left to _mic_tick. That fragment only renders inside the `turn is not
    None and not waiting` branch, so the moment the bridge goes up it stops running
    entirely — while the MicSession, the WS server thread and push_frame all live outside
    Streamlit and carry right on. The result was that talking over the bridge got queued
    and then drained by the NEXT turn's first tick, which submitted it as that turn's
    reaction and pauses it at pos≈0.
    """
    sess.mic_armed = False
    mic_session = st.session_state.get("mic_session")
    if mic_session is not None:
        mic_session.disarm()


def _arm_mic_session(sess: InteractiveSession) -> None:
    """
    Re-arm once the new turn's audio is up (§1.6). Everything captured while disarmed is
    dropped here — a queued mid-bridge utterance and a stale onset timestamp are both from
    a turn that is already over, so neither may surface against the new one.
    """
    sess.arm_mic()
    mic_session = st.session_state.get("mic_session")
    if mic_session is None:
        return
    while True:
        try:
            mic_session.utterance_queue.get_nowait()
        except queue.Empty:
            break
    mic_session.last_seen_onset_at = mic_session.last_onset_at
    mic_session.arm()


def _start_reaction(sess: InteractiveSession, reaction_text: str, anchor_snippet: str) -> None:
    _disarm_mic_session(sess)
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


def _get_or_create_mic_session(sess: InteractiveSession):
    from personacast.pipeline import mic_transport_ws
    from personacast.pipeline.mic import MicSession

    existing = st.session_state.get("mic_session")
    if existing is not None:
        mic_transport_ws.set_active_session(existing)  # cheap, idempotent; safe every render
        return existing

    utterance_queue: queue.Queue = queue.Queue()

    def on_speech_start() -> None:
        # off-thread (WS server thread) -- must be non-blocking, never touch st.session_state
        mic_session.last_onset_at = time.monotonic()

    def on_utterance(wav_bytes: bytes) -> None:
        # off-thread (WS server thread) -- push onto the thread-safe queue, no Streamlit calls here
        utterance_queue.put(wav_bytes)

    mic_session = MicSession(on_speech_start, on_utterance)
    mic_session.utterance_queue = utterance_queue
    mic_session.last_onset_at = None
    mic_session.frames_seen = 0
    mic_session.last_frame_at = None
    mic_session.last_seen_onset_at = None
    st.session_state["mic_session"] = mic_session
    mic_transport_ws.set_active_session(mic_session)
    return mic_session


@st.fragment(run_every=0.25)
def _mic_tick(sess: InteractiveSession, turn, speak: bool) -> None:
    from personacast.pipeline.mic import MicState

    mic_session = st.session_state.get("mic_session")
    if mic_session is None:
        return

    ### rendered here (not the outer script body) specifically so pos_{turn.iteration}
    ### stays fresh every 0.25s instead of freezing at its first-render value -- see
    ### _render_turn_audio's docstring for the full explanation of why that mattered.
    if speak:
        _render_turn_audio(sess, turn)

    ### the mic status indicator itself lives in the injected browser-side HTML
    ### (_MIC_CAPTURE_HTML's #pc-mic-status div, right above the mic toggle) -- it already
    ### reports connecting/streaming/permission-denied with the actual browser error text,
    ### so there's no separate Python-side status line here to avoid showing the same
    ### thing twice.

    ### sync MicSession's internal state to sess.mic_armed (set false the instant
    ### begin_reaction fires, true again once the next turn's audio is set -- see arm_mic()
    ### call sites). best-effort: this sync runs on the script thread once per 0.25s tick
    ### while push_frame runs on the audio thread; a production build would want a lock here.
    if sess.mic_armed and mic_session.state == MicState.DISARMED:
        mic_session.arm()
    elif not sess.mic_armed and mic_session.state != MicState.DISARMED:
        mic_session.disarm()

    ### onset-based anchor capture + ACTUAL pause: snapshot the pause position the instant
    ### we notice a fresh onset, replacing today's button-click-time read (up to ~0.25s slop
    ### vs the true onset instant -- sanity-check against the Step 0 spike's measured
    ### latency). audix has no imperative pause control (checked its source directly --
    ### only mount-time autoplay/start_time props), so "pausing" means forcing a fresh
    ### remount at this position with autoplay=False; see _play's playback_override.
    if mic_session.last_onset_at is not None and mic_session.last_onset_at != mic_session.last_seen_onset_at:
        mic_session.last_seen_onset_at = mic_session.last_onset_at
        ### pos_{iteration} may still be None the very first time someone speaks right
        ### after a turn starts (audix hasn't reported a currentTime yet) -- fall back to
        ### 0.0 rather than silently skipping the pause. missing this fallback meant the
        ### first onset in a turn got marked "handled" without ever actually pausing.
        pos = st.session_state.get(f"pos_{turn.iteration}") or 0.0
        st.session_state[f"mic_anchor_pos_{turn.iteration}"] = pos
        st.session_state[f"mic_playback_override_{turn.iteration}"] = (pos, False)
        ### §1.9's acceptance check is onset->pause < ~300ms, and it has never actually been
        ### measured. This is the Python-side half of that budget (VAD onset -> pause state
        ### committed); the audix remount that follows is unmeasured from here. Recording it
        ### rather than tuning blind -- the levers (ScriptProcessorNode buffer size,
        ### run_every) are cheap but the remount is likely the dominant term.
        st.session_state["mic_onset_to_pause_ms"] = (
            time.monotonic() - mic_session.last_onset_at) * 1000.0
        st.rerun(scope="app")
        return

    try:
        wav_bytes = mic_session.utterance_queue.get_nowait()
    except queue.Empty:
        return

    reaction_text = stt.transcribe(wav_bytes)
    if not reaction_text.strip():
        mic_session.arm()  # empty transcript -> silent resume, no reaction
        resume_pos = st.session_state.get(f"mic_anchor_pos_{turn.iteration}")
        if resume_pos is not None:
            st.session_state[f"mic_playback_override_{turn.iteration}"] = (resume_pos, True)
        st.rerun(scope="app")
        return

    st.session_state["heard"] = reaction_text
    anchor_pos = st.session_state.get(f"mic_anchor_pos_{turn.iteration}")
    path = st.session_state.get("response_audio")
    dur = tts.wav_duration(path) if path else 0.0
    anchor_snippet = ""
    ### anchor_pos can legitimately be 0.0 (interrupted right at the start of a turn, or
    ### before audix had reported any position yet -- see the onset handler's "or 0.0"
    ### fallback above). `if anchor_pos and dur` treats that valid 0.0 as falsy and
    ### silently skips the anchor -- exactly the case that was never registering one.
    if anchor_pos is not None and dur:
        anchor_snippet = interactive_mod.locate_snippet(turn.text, anchor_pos, dur)

    st.session_state.pop(f"mic_playback_override_{turn.iteration}", None)
    _start_reaction(sess, reaction_text, anchor_snippet)
    st.rerun(scope="app")


### Static (byte-identical across reruns, aside from the port) browser-side mic capture.
### Opens a native WebSocket directly to mic_transport_ws's server -- no Streamlit
### component protocol involved in the actual audio transport, which is the whole point:
### that protocol's rerun-driven teardown was the root cause of the streamlit-webrtc
### instability. ScriptProcessorNode is deprecated but chosen deliberately for its
### simplicity (no separate AudioWorklet module file needed) -- acceptable for a local
### research tool. The silent-gain node keeps the audio graph alive without audible
### passthrough (no echo/feedback to the speakers).
_MIC_CAPTURE_HTML = """
<div id="pc-mic-status" style="font-family: sans-serif; font-size: 13px; color: #888;">
  🟡 requesting mic permission…
</div>
<script>
(function() {
  if (window.__personacastMicStarted) { return; }
  window.__personacastMicStarted = true;

  // the div above MUST come before this <script> tag in source order -- inline scripts
  // execute immediately as the parser reaches them, so getElementById would return null
  // (and every setStatus() call below would silently no-op forever, exactly as it did
  // before this fix: the status text stayed frozen on the placeholder even while the
  // underlying getUserMedia/WebSocket pipeline was actually working correctly, confirmed
  // directly via a headless-browser repro with a fake mic device -- frames were flowing
  // into Python the whole time, only the on-page indicator was silently broken).
  var status = document.getElementById("pc-mic-status");
  function setStatus(text) { if (status) status.textContent = text; }

  // autoGainControl/noiseSuppression off deliberately: browsers' AGC is known to cause
  // audible pumping/distortion right as it adjusts to a sudden loud sound -- exactly what
  // happens at the start of every interruption, matching reported "clipping" in captures.
  // echoCancellation stays on as a cheap safety net even with headphones required.
  navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: false,
      autoGainControl: false
    },
    video: false
  }).then(function(stream) {
    var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    var source = audioCtx.createMediaStreamSource(stream);
    var processor = audioCtx.createScriptProcessor(4096, 1, 1);
    var silentGain = audioCtx.createGain();
    silentGain.gain.value = 0;

    var ws = null;
    function connect() {
      ws = new WebSocket("ws://localhost:__PORT__");
      ws.binaryType = "arraybuffer";
      ws.onopen = function() {
        ws.send(JSON.stringify({sampleRate: audioCtx.sampleRate}));
        setStatus("🟢 mic streaming to PersonaCast");
      };
      ws.onclose = function() { setStatus("🟡 mic disconnected, retrying…"); setTimeout(connect, 1000); };
      ws.onerror = function() { try { ws.close(); } catch (e) {} };
    }
    connect();

    processor.onaudioprocess = function(e) {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      var input = e.inputBuffer.getChannelData(0);
      var pcm16 = new Int16Array(input.length);
      for (var i = 0; i < input.length; i++) {
        var s = Math.max(-1, Math.min(1, input[i]));
        pcm16[i] = s < 0 ? s * 32768 : s * 32767;
      }
      ws.send(pcm16.buffer);
    };

    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioCtx.destination);
    setStatus("🟡 mic connecting…");
  }).catch(function(err) {
    setStatus("🔴 mic permission denied or unavailable: " + err.message);
  });
})();
</script>
"""


def _render_always_on_mic(sess: InteractiveSession, turn, speak: bool) -> None:
    try:
        import torch  # noqa: F401
        import websockets  # noqa: F401
    except ImportError:
        st.warning(
            "MIC_ALWAYS_ON is on but websockets/torch aren't installed (see requirements.txt)."
        )
        return

    from personacast.pipeline import mic_transport_ws

    st.caption("⚠️ Headphones required — no echo cancellation in v1.")
    _get_or_create_mic_session(sess)  # created once, reused across reruns
    try:
        port = mic_transport_ws.ensure_server_running()
    except RuntimeError as err:
        ### say what's actually wrong instead of injecting mic JS that can only fail and
        ### leave the user staring at "mic disconnected, retrying…" forever.
        st.error(f"🎙️ {err}")
        return
    st.components.v1.html(_MIC_CAPTURE_HTML.replace("__PORT__", str(port)), height=40)
    _mic_tick(sess, turn, speak)


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

        if session.bank_status:
            st.caption(f"⚠️ {session.bank_status}")
        onset_ms = st.session_state.get("mic_onset_to_pause_ms")
        if onset_ms is not None:
            st.caption(f"⏱ last interruption: {onset_ms:.0f}ms from speech onset to pause")

        turn = st.session_state.get("current_turn")
        waiting = st.session_state.get("pending_future") is not None

        if waiting:
            opener = st.session_state.get("opener", {})
            st.subheader("🔊 Bridging…")
            st.write(opener.get("text", ""))
            _play(st.session_state.get("bridge_audio"), f"bridge_{turn.iteration}", "🌉 Bridging…")
            _playback_tick(session)

        elif turn is not None:
            words = len(turn.text.split())
            st.subheader(f"Turn {turn.iteration} · {turn.topic}")

            n_sources = len(session.state.pool.get(turn.topic, []))
            st.caption(f"~{words} words · ~{words / 155 * 60:.0f}s · {n_sources} sources in pool")
            st.write(turn.text)

            mic_active = st.session_state.get("mic_always_on", config.MIC_ALWAYS_ON)
            if speak and not mic_active:
                ### mic_active turns render this INSIDE _mic_tick's 0.25s fragment instead
                ### (see _render_turn_audio) -- see that function's docstring for why.
                _render_turn_audio(session, turn)

            path = st.session_state.get("response_audio")
            pos = st.session_state.get(f"pos_{turn.iteration}")
            dur = tts.wav_duration(path) if path else 0.0
            sentences = _split_sentences(turn.text)
            _whole = "(react to the whole turn)"
            options = [_whole] + sentences
            akey = f"anchor_{turn.iteration}"
            if pos is not None and dur:
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

            if st.session_state.get("mic_always_on", config.MIC_ALWAYS_ON):
                _render_always_on_mic(session, turn, speak)

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
                ### finish() now makes one LLM call to revise persona_style from this
                ### session's reactions, so it blocks for a few seconds
                with st.spinner("Wrapping up — learning your delivery preferences…"):
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

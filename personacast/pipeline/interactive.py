"""
the interactive session orchestrator

retrieve + curate ipelie runs once, we synthesize from the resulting pool
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from .. import config
from ..agents import interaction
from ..llm.client import BudgetExceeded, LLMClient
from ..models import (
    CuratedItem,
    InteractiveTurn,
    Persona,
    Reaction,
    ReactionType,
    SessionState,
    PersonaMemory,
    CoveredSource
)
from . import memory, openers as openers_mod, poolcache, qa, script, state, timing, tts
from .retrieval.tavily import search_web

def build_source_pool(persona: Persona, llm: LLMClient, on_stage = None,
                       *, memory: PersonaMemory | None = None, on_topic_done = None) -> dict[str, list[CuratedItem]]:
    from ..agents.graph import build_source_pool_agentic

    return build_source_pool_agentic(
        persona, llm, on_stage=on_stage, memory=memory, on_topic_done=on_topic_done
    )




_GIST_SYSTEM = (
    "Condense the following podcast turn into ONE short plain sentence capturing only the key "
    "point(s) it covered — it will be used as a 'what we already said' note so later turns don't "
    "repeat it. No preamble, no quotes, just the sentence."
)


def _first_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[0] if sentences and sentences[0] else text[:160]



def  summarize_turn(text: str, llm: LLMClient):
    if not config.SUMMARIZE_TURNS:
        return _first_sentence(text)

    try:
        return llm.complete(_GIST_SYSTEM, text, temperature=0.2, priority="background").strip()
    except BudgetExceeded:
        return _first_sentence(text)


_SUMMARY_MAX_LINES = 12 # how many lines of topic gist the cross session summary can keep 

def _update_summary(summary:str, turn: InteractiveTurn): 
    """
    append turn gist to rolling summary, capped at _SUMMARY_MAX_LINES
    """
    lines = [line for line in summary.splitlines() if line.strip()]
    lines.append(f"- [{turn.topic}] {turn.gist}")
    return "\n".join(lines[-_SUMMARY_MAX_LINES:])

### mapping pause time to sentence being spoken

def locate_snippet(text, seconds:float, total_seconds:float) -> str: 
    """
    given the audio pause time, find the sentence podcast was at
    seconds/total -> word index -> sentence containing word returns "" if nothing
    """

    text = (text or "").strip() # given full turn text  remove surrounding whitespace 
    if not text or total_seconds <= 0: #if no text return nothing
            return ""

    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()] # split sentences into list elements

    if not sentences: # another guard if splitting created nothing
        return ""

    fraction = min(max(seconds / total_seconds, 0.0), 1.0) # how far through audio the pause has happened as fraction if halfway through text 0.5

    target_word_position = fraction * len(text.split()) #estimated word number position 

    seen_word_count = 0 # accumulating word count
    ## logic: target word is at 5 word ocount psition, sentence A -> 4 words seen = 4, 5< 4 dalse move sentence B -> 3 words seen = 7 5 <7  true in sentence B!

    for sentence in sentences: 
        seen_word_count += len(sentence.split())
        if target_word_position < seen_word_count: 
            return sentence.strip()

    return sentences[-1].strip() # return sentence target word is in so we know sentence interrupted at 


### given the anchoring sentence make llm call to find the curated source that is associated with what that sentence is talking abt

_WH_WORDS = {
    "what", "why", "how", "when", "where", "who", "whose", "which",
    "is", "are", "was", "were", "does", "do", "did", "can", "could",
    "would", "should", "will", "isnt", "doesnt",
}

def _looks_like_question(text: str) -> bool:
    stripped = (text or "").strip().lower()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    return stripped.split()[0].strip(",.") in _WH_WORDS


@contextmanager
def _null():
    yield


def  _record_covered(session_state: SessionState, topic: str, sources: list[CuratedItem]) -> None:
    shown_sources_memory = session_state.memory.covered.setdefault(topic, [])
    seen = {source.url for source in shown_sources_memory}

    for item in sources:
        if item.url not in seen:
            seen.add(item.url)
            shown_sources_memory.append(CoveredSource(url= item.url, title= item.title))


class InteractiveSession:

    def __init__(self, persona: Persona, llm: LLMClient |None = None,*, on_stage = None):
        self.persona = persona
        self.llm = llm or LLMClient(on_stage=on_stage)
        self.on_stage = on_stage
        self.state: SessionState | None = None
        self._active_topics = [interest.topic for interest in persona.interests]

        self.retrieval_trace: dict[str, dict] = {}

        self.pool_from_cache = False

        self._workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pc-inner")
        self._continuation = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pc-turn")

        self.openers: dict[int, dict[str, str]] = {}
        self.opener_audio: dict[int, dict[str, str]] = {}
        self._opener_jobs: dict[int, object] = {}

        self.timings: dict[int, dict] = {}

        self.audio_chunks: list[dict] = []

    def _note_topic(self, topic: str, final_state: dict) -> None:
        self.retrieval_trace[topic] = {
            "sources": final_state.get("sources", []),
            "queries": final_state.get("search_queries", []),
            "arxiv_queries": final_state.get("arxiv_queries", []),
            "retrieved": len(final_state.get("raw_items", [])),
            "kept": len(final_state.get("curated", [])),
            "notes": final_state.get("notes", []),
        }


    def start(self, *, rebuild_pool: bool = False) -> SessionState:
        mem = memory.load_memory(self.persona)

        pool = None if rebuild_pool else poolcache.load(self.persona)
        if pool is not None:
            self.pool_from_cache = True
            if self.on_stage:
                age = poolcache.age_hours(self.persona) or 0.0
                self.on_stage(f"Reusing cached source pool ({age:.1f}h old) — no retrieval calls")
        else:
            pool = build_source_pool(self.persona, self.llm, on_stage= self.on_stage, memory = mem, on_topic_done= self._note_topic)
            poolcache.save(self.persona, pool)

        self.state = SessionState(
            run_id = state.new_run_id(), persona = self.persona, memory = mem, pool = pool
        )

        if self.retrieval_trace:
            state.log_retrieval(self.state.run_id, self.retrieval_trace)

        self._warm_static_openers()

        return self.state

    @property
    def done(self) -> bool:
        return self.state is not None and len(self.state.turns) >= config.MAX_ITERATIONS

    def _choose_topic(self, session_state : SessionState, last_reaction : Reaction | None):

        if last_reaction and last_reaction.requested_topic in self._active_topics:
            return last_reaction.requested_topic

        if session_state.current_topic is None:
            return memory.next_topic(session_state.memory, self._active_topics)

        if last_reaction and last_reaction.type == ReactionType.none:
            best = memory.next_topic(session_state.memory, self._active_topics)

            if session_state.memory.engagement.get(best, config.ENGAGE_BASE)  > session_state.memory.engagement.get(session_state.current_topic, config.ENGAGE_BASE):
                return best

            i = self._active_topics.index(session_state.current_topic)
            return self._active_topics[(i + 1) % len(self._active_topics)]

        return session_state.current_topic

    def _require_started(self) -> SessionState:
        if self.state is None:
            raise RuntimeError("call start() before generating turns")
        return self.state

    def _focus_source(self, session_state: SessionState, last_reaction: Reaction | None) -> CuratedItem | None:
        if last_reaction is None or last_reaction.anchor_source_index < 0:
            return None

        sources = session_state.pool.get(last_reaction.topic, [])
        index = last_reaction.anchor_source_index
        return sources[index] if index < len(sources) else None

    def _turn_context(self, session_state: SessionState):
        last_reaction = session_state.turns[-1].reaction if session_state.turns else None
        topic = self._choose_topic(session_state, last_reaction)
        session_state.current_topic = topic
        recent_gists = [t.gist for t in session_state.turns[-config.RECENT_TURNS_CONTEXT:] if t.gist]
        focus_source = self._focus_source(session_state, last_reaction)
        return last_reaction, topic, recent_gists, focus_source, session_state.pool.get(topic, [])

    def _finalize_turn(self, session_state: SessionState, topic: str, text: str,
                       sources: list[CuratedItem]) -> InteractiveTurn:
        turn = InteractiveTurn(iteration=len(session_state.turns) + 1, topic=topic, text=text)
        session_state.turns.append(turn)
        _record_covered(session_state, topic, sources)
        self.prepare_openers(turn)
        return turn

    def publish(self, kind: str, text: str, path: str | None) -> None:
        self.audio_chunks.append({"kind": kind, "text": text, "path": path})

    def next_segment(self, *, opener_text: str = ""):
        session_state = self._require_started()
        last_reaction, topic, recent_gists, focus_source, sources = self._turn_context(session_state)

        text = script.generate_turn(
            topic, sources, self.persona, session_state.memory, recent_gists, self.llm,
            last_reaction=last_reaction, focus_source= focus_source, opener_text=opener_text,
        )
        return self._finalize_turn(session_state, topic, text, sources)

    def generate_segment(self, *, opener_text: str = "", timer=None) -> InteractiveTurn:
        session_state = self._require_started()
        _last, topic, gists, focus, sources = self._turn_context(session_state)
        iteration = len(session_state.turns) + 1
        out_dir = state.run_dir(session_state.run_id)

        text = script.generate_turn(
            topic, sources, self.persona, session_state.memory, gists, self.llm,
            last_reaction=_last, focus_source=focus, opener_text=opener_text,
        )

        with (timer.stage(timing.TTS) if timer else _null()):
            try:
                path = str(tts.synthesize(text, out_dir / f"turn_{iteration}.wav"))
            except Exception:
                path = None

        self.publish("turn", text, path)
        if timer is not None:
            timer.mark("turn_audio_ready")

        return self._finalize_turn(session_state, topic, text, sources)


    def _likely_switch_topic(self, current_topic: str) -> str | None:
        others = [t for t in self._active_topics if t != current_topic]
        if not others:
            return None
        return memory.next_topic(self.state.memory, others)

    def prepare_openers(self, turn: InteractiveTurn) -> None:
        if turn.iteration in self._opener_jobs:
            return

        def build():
            pool = openers_mod.generate_openers(
                turn, self.persona, self.llm,
                likely_switch_topic=self._likely_switch_topic(turn.topic),
            )
            self.openers[turn.iteration] = pool
            self.opener_audio[turn.iteration] = openers_mod.synthesize_openers(
                pool, state.run_dir(self.state.run_id), turn.iteration,
                on_error=self.on_stage,
            )

        self._opener_jobs[turn.iteration] = self._workers.submit(build)

    def _warm_static_openers(self) -> None:
        try:
            self.opener_audio[0] = openers_mod.synthesize_openers(
                openers_mod.fallback_openers(), state.run_dir(self.state.run_id), 0,
                on_error=self.on_stage,
            )
        except Exception:
            self.opener_audio[0] = {}

    def begin_reaction(self, reaction_text: str, *, anchor_snippet: str = ""):
        session_state = self._require_started()
        if not session_state.turns:
            raise RuntimeError("begin_reaction called before next_segment")

        turn = session_state.turns[-1]
        timer = timing.Timer()
        self.audio_chunks = []

        with timer.stage(timing.OPENER_SELECT):
            name, text, wav = openers_mod.select_opener(
                self.openers.get(turn.iteration, {}),
                self.opener_audio.get(turn.iteration) or self.opener_audio.get(0, {}),
                reaction_text, self._active_topics, turn.topic,
            )

        if wav:
            timer.mark(timing.FIRST_AUDIO)

        future = self._continuation.submit(
            self._continue_after, reaction_text, anchor_snippet, text, timer,
        )
        return {"class": name, "text": text, "audio": wav}, future

    def _continue_after(self, reaction_text: str, anchor_snippet: str, opener_text: str,
                        timer: timing.Timer | None = None):
        reacted_to = self.state.turns[-1].iteration
        self.submit_reaction(reaction_text, anchor_snippet=anchor_snippet, timer=timer)


        if self.done:
            self.finish()
            result = None
        else:
            with (timer.stage(timing.GENERATE) if timer else _null()):
                result = self.generate_segment(opener_text=opener_text, timer=timer)

        if timer is not None:
            timer.mark("continuation_ready")
            self.timings[reacted_to] = timer.as_dict()
            if self.on_stage:
                self.on_stage(f"⏱ {timer.summary()}")
        return result

    def _resolve_switch(self, plan, current_topic: str) -> str | None:
        requested = (plan.requested_topic or "").strip() or None
        if requested == current_topic or requested not in self._active_topics:
            return None
        return requested

    def _web_fallback(self, plan, reaction_text: str, web_future) -> tuple[str, bool]:
        if web_future is None:
            return plan.answer, False

        try:
            web_items = web_future.result(timeout=config.WEB_FALLBACK_TIMEOUT_SECONDS)
        except Exception:
            return plan.answer, False

        if not web_items:
            return plan.answer, False

        try:
            web_answer = interaction.answer_from_web(
                reaction_text, self.persona, qa._web_to_curated(web_items), self.llm,
            )
        except Exception:
            return plan.answer, False

        if not web_answer.answered or not web_answer.answer.strip():
            return plan.answer, False

        return web_answer.answer, True

    def submit_reaction(self, reaction_text: str, *, anchor_snippet: str = "",
                        timer: timing.Timer | None = None):

        session_state = self._require_started()

        if not session_state.turns:
            raise RuntimeError("submit_reaction called before next_segment")

        turn = session_state.turns[-1]

        topic_sources = session_state.pool.get(turn.topic, [])

        llm_before = self.llm.snapshot()

        plan_future = self._workers.submit(
            interaction.interpret,
            reaction_text, turn, session_state, self.persona, self.llm,
            active_topics=self._active_topics, sources=topic_sources,
            anchor_snippet=anchor_snippet.strip(),
        )
        web_future = None
        if _looks_like_question(reaction_text):
            web_future = self._workers.submit(
                search_web, reaction_text, topic="general", days=None,
            )

        with (timer.stage(timing.INTERPRET) if timer else _null()):
            plan = plan_future.result()
        if timer is not None:
            timer.marks.update({f"llm_{k}": v for k, v in self.llm.since(llm_before).items()})

        requested = self._resolve_switch(plan, turn.topic)
        reaction_type = interaction.to_reaction_type(plan.intent)
        delta = interaction.clamp_delta(plan.engagement_delta, is_switch=requested is not None)

        anchor = "" if reaction_type == ReactionType.none else anchor_snippet.strip()
        anchor_index = plan.anchor_source_index if anchor else -1
        if not (0 <= anchor_index < len(topic_sources)):
            anchor_index = -1

        reaction = Reaction(
            iteration=turn.iteration, topic=turn.topic, type=reaction_type,
            text=reaction_text.strip(), requested_topic=requested, answer=None,
            anchor_snippet=anchor, anchor_source_index=anchor_index,
            anchor_source=topic_sources[anchor_index].title if anchor_index >= 0 else "",
            intent=plan.intent, sentiment=plan.sentiment, engagement_delta=delta,
        )

        if plan.needs_answer:
            if plan.answered:
                reaction.answer = plan.answer
            else:
                with (timer.stage(timing.WEB_ANSWER) if timer else _null()):
                    reaction.answer, reaction.used_web = self._web_fallback(
                        plan, reaction.text, web_future,
                    )

        if web_future is not None and not web_future.done():
            web_future.cancel()

        turn.reaction = reaction

        memory.apply_reaction(session_state.memory, reaction, delta=delta)

        turn.gist = plan.gist.strip() or summarize_turn(turn.text, self.llm)
        session_state.memory.summary = _update_summary(session_state.memory.summary, turn)

        memory.save_memory(session_state.memory)
        state.log_turn(session_state, turn.iteration)
        return turn

    def finish(self) -> SessionState:
        session_state = self._require_started()
        memory.save_memory(session_state.memory)
        transcript = "\n\n".join(f"[{t.topic}] {t.text}" for t in session_state.turns)
        (state.run_dir(session_state.run_id) / "session.txt").write_text(transcript)

        if self.timings:
            state.log_timings(session_state.run_id, self.timings)

        self._workers.shutdown(wait=False, cancel_futures=True)
        self._continuation.shutdown(wait=False, cancel_futures=True)

        return session_state

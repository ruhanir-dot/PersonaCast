"""
the interactive session orchestrator

retrieve + curate ipelie runs once, we synthesize from the resulting pool
"""

from __future__ import annotations

import json
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


def load_trunk_pool(on_stage = None, *, persona_id: str | None = None):
    if not config.TRUNKS:
        return None

    from ..trunks import store as trunk_store

    try:
        loaded = trunk_store.load(persona_id=persona_id)
    except trunk_store.TrunkStoreError as err:
        if on_stage:
            on_stage(f"Trunk pool unavailable, falling back to live retrieval — {err}")
        return None

    return loaded




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

        self.trunks = None
        self.current_trunks: dict[int, object] = {}

        self.first_topic: str | None = None

        self._workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pc-inner")
        self._continuation = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pc-turn")

        ### add in bridge bank attributes
        self.bridge_bank: openers_mod.BridgeBank | None = None
        self.bridge_audio: dict[str, dict[int, str]] = {name: {} for name in openers_mod.OPENER_CLASSES}
        self._fallback_audio: dict[str, str] = {}
        self._bank_job = None
        self.recent_bridge_picks: dict[str, list[int]] = {}
        self._style_updated = False
        self.bank_status: str | None = None

        self.mic_armed: bool = False

        self.timings: dict[int, dict] = {}

        self.audio_chunks: list[dict] = []

    def _align_topics_to_pool(self, pool: dict[str, list[CuratedItem]]) -> None:
        pool_topics = list(pool.keys())
        if not pool_topics:
            return

        persona_topics = [interest.topic for interest in self.persona.interests]
        unmatched = [topic for topic in persona_topics if topic not in pool]

        self._active_topics = pool_topics

        if unmatched and self.on_stage:
            self.on_stage(
                f"Persona topics not in the trunk pool ({', '.join(unmatched[:3])}"
                f"{'…' if len(unmatched) > 3 else ''}) — using the pool's topics instead"
            )

    def _warm_trunk_encoder(self) -> None:
        if self.trunks is None:
            return
        from ..trunks import embed as trunk_embed

        self._workers.submit(trunk_embed.warm)

    def _note_topic(self, topic: str, final_state: dict) -> None:
        self.retrieval_trace[topic] = {
            "sources": final_state.get("sources", []),
            "queries": final_state.get("search_queries", []),
            "arxiv_queries": final_state.get("arxiv_queries", []),
            "retrieved": len(final_state.get("raw_items", [])),
            "kept": len(final_state.get("curated", [])),
            "notes": final_state.get("notes", []),
        }


    def start(self, *, rebuild_pool: bool = False, first_topic: str | None = None) -> SessionState:
        if first_topic:
            self.first_topic = first_topic.strip() or None

        mem = memory.load_memory(self.persona)
        memory.seed_persona_style_if_needed(mem, self.persona, self.llm) #seed in the persona style vector at persona construction if cold user

        self.trunks = load_trunk_pool(on_stage=self.on_stage,
                                      persona_id=self.persona.persona_id)

        if self.trunks is not None:
            pool = self.trunks.pool.as_source_pool(summary_chars=config.TRUNK_SUMMARY_CHARS)
            self.pool_from_cache = True
            self._align_topics_to_pool(pool)
            if self.first_topic and self.first_topic not in self._active_topics:
                if self.on_stage:
                    self.on_stage(
                        f"Requested opening topic '{self.first_topic}' is not in the pool — "
                        "starting on the highest-engagement topic instead"
                    )
                self.first_topic = None
            if self.on_stage:
                self.on_stage(
                    f"Using offline trunk pool — {len(self.trunks.pool.topics)} topics, "
                    f"{len(self.trunks.pool.all_trunks())} candidate segments, no retrieval calls"
                )
        else:
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

        self._warm_fallback_bridges() # pre synthesize the fallback bridge
        self._start_bank_generation() # one time bank generation
        self._warm_trunk_encoder() # so the first interruption isn't the one paying the model load

        return self.state

    @property
    def done(self) -> bool:
        return self.state is not None and len(self.state.turns) >= config.MAX_ITERATIONS

    def _choose_topic(self, session_state : SessionState, last_reaction : Reaction | None):

        if last_reaction and last_reaction.requested_topic in self._active_topics:
            return last_reaction.requested_topic

        if session_state.current_topic is None:
            if self.first_topic and self.first_topic in self._active_topics:
                return self.first_topic
            return memory.next_topic(session_state.memory, self._active_topics)

        if last_reaction and last_reaction.type == ReactionType.none:
            best = memory.next_topic(session_state.memory, self._active_topics)

            if session_state.memory.engagement.get(best, config.ENGAGE_BASE)  > session_state.memory.engagement.get(session_state.current_topic, config.ENGAGE_BASE):
                return best

            try:
                i = self._active_topics.index(session_state.current_topic)
            except ValueError:
                return best
            return self._active_topics[(i + 1) % len(self._active_topics)]

        if session_state.current_topic not in self._active_topics:
            return memory.next_topic(session_state.memory, self._active_topics)

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

    def _select_trunk(self, session_state: SessionState, topic: str):
        if self.trunks is None:
            return None

        from ..trunks import select as trunk_select

        picked = trunk_select.next_trunk(self.trunks.pool, session_state, topic)
        if picked is None:
            return None

        trunk, _index = picked
        self.current_trunks[len(session_state.turns) + 1] = trunk
        return trunk

    def _turn_context(self, session_state: SessionState):
        last_reaction = session_state.turns[-1].reaction if session_state.turns else None

        if self.trunks is not None:
            from ..trunks import select as trunk_select

            self._active_topics = trunk_select.live_topics(
                self.trunks.pool, session_state, self._active_topics
            )

        topic = self._choose_topic(session_state, last_reaction)
        session_state.current_topic = topic
        recent_gists = [t.gist for t in session_state.turns[-config.RECENT_TURNS_CONTEXT:] if t.gist]
        focus_source = self._focus_source(session_state, last_reaction)

        sources = session_state.pool.get(topic, [])
        trunk = self._select_trunk(session_state, topic)

        return last_reaction, topic, recent_gists, focus_source, sources, trunk

    def _finalize_turn(self, session_state: SessionState, topic: str, text: str,
                       sources: list[CuratedItem], trunk=None) -> InteractiveTurn:
        turn = InteractiveTurn(iteration=len(session_state.turns) + 1, topic=topic, text=text)
        session_state.turns.append(turn)

        if trunk is not None:
            covered = [trunk.to_curated_item(summary_chars=config.TRUNK_SUMMARY_CHARS)]
        else:
            covered = sources

        _record_covered(session_state, topic, covered)
        return turn

    def publish(self, kind: str, text: str, path: str | None) -> None:
        self.audio_chunks.append({"kind": kind, "text": text, "path": path})

    def next_segment(self, *, opener_text: str = ""):
        session_state = self._require_started()
        last_reaction, topic, recent_gists, focus_source, sources, trunk = self._turn_context(session_state)

        text = script.generate_turn(
            topic, sources, self.persona, session_state.memory, recent_gists, self.llm,
            last_reaction=last_reaction, focus_source= focus_source, opener_text=opener_text,
            draft=trunk.script if trunk else "",
            draft_focus=trunk.focus if trunk else "",
        )
        return self._finalize_turn(session_state, topic, text, sources, trunk)

    def generate_segment(self, *, opener_text: str = "", timer=None) -> InteractiveTurn:
        session_state = self._require_started()
        _last, topic, gists, focus, sources, trunk = self._turn_context(session_state)
        iteration = len(session_state.turns) + 1
        out_dir = state.run_dir(session_state.run_id)

        text = script.generate_turn(
            topic, sources, self.persona, session_state.memory, gists, self.llm,
            last_reaction=_last, focus_source=focus, opener_text=opener_text,
            draft=trunk.script if trunk else "",
            draft_focus=trunk.focus if trunk else "",
        )

        with (timer.stage(timing.TTS) if timer else _null()):
            try:
                path = str(tts.synthesize(text, out_dir / f"turn_{iteration}.wav"))
            except Exception:
                path = None

        self.publish("turn", text, path)
        if timer is not None:
            timer.mark("turn_audio_ready")

        return self._finalize_turn(session_state, topic, text, sources, trunk)


    def _start_bank_generation(self) -> None:
        """
        bridge bank generation once per session thy are persona and context based and dont rely on turns 
        """
        def report(msg: str) -> None:
            self.bank_status = msg
            if self.on_stage:
                self.on_stage(msg)

        def build():
            # building the candidate bank
            try:

                bank = openers_mod.generate_bridge_bank(
                    self.persona, self.persona.additional_context, self.llm,
                    active_topics=self._active_topics,
                    memory_summary=self.state.memory.summary,
                    on_error=report,
                )

                self.bridge_bank = bank

                total = sum(len(bank.by_class(n)) for n in openers_mod.OPENER_CLASSES) # check total amount generated

                if total == 0: # fallback if llm call failed and empty 
                    if self.bank_status is None:
                        report("bridge bank came back empty — using static fallback bridges "
                               "this session")
                    return
                
                openers_mod.synthesize_bridge_bank(
                    bank, state.run_dir(self.state.run_id), self.bridge_audio,
                    on_error=report,
                )

            except Exception as err:
                report(f"bridge bank build failed: {type(err).__name__}: {err} — "
                       "using static fallback bridges this session")

        self._bank_job = self._workers.submit(build)

    def _warm_fallback_bridges(self) -> None:
        """
        pre synthesize the static fallback bridges if llm call failed or if user interrupts before bridge bank generated
        """
        out_dir = state.run_dir(self.state.run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        fallback = openers_mod.fallback_openers()
        paths: dict[str, str] = {}
        for name, text in fallback.items():
            try:
                target = out_dir / f"bridge_fallback_{name}.wav"
                paths[name] = str(tts.synthesize(text, target))
            except Exception as err:
                if self.on_stage:
                    self.on_stage(f"fallback bridge synthesis failed for {name}: {type(err).__name__}")
        self._fallback_audio = paths

    def arm_mic(self) -> None:
        self.mic_armed = True

    def begin_reaction(self, reaction_text: str, *, anchor_snippet: str = ""):
        session_state = self._require_started()
        if not session_state.turns:
            raise RuntimeError("begin_reaction called before next_segment")

        self.mic_armed = False

        turn = session_state.turns[-1]
        timer = timing.Timer()
        self.audio_chunks = []

        with timer.stage(timing.OPENER_SELECT): # timer for measuring latency

            name, text, wav, index =  openers_mod.select_opener(
                self.bridge_bank, self.bridge_audio, self._fallback_audio,
                session_state.memory.persona_style,
                reaction_text, self._active_topics, turn.topic,
                recent_indices=self.recent_bridge_picks,
            )
            if index is not None:
                window = max(0, config.BRIDGE_RECENCY_WINDOW)
                recent = self.recent_bridge_picks.setdefault(name, [])
                recent.append(index)
                del recent[: max(0, len(recent) - window)]

        if wav:
            timer.mark(timing.FIRST_AUDIO)

        future = self._continuation.submit(
            self._continue_after, reaction_text, anchor_snippet, text, timer,
        )
        return {"class": name, "text": text, "audio": wav}, future

    def _continue_after(self, reaction_text: str, anchor_snippet: str, opener_text: str, timer: timing.Timer | None = None):

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

    def _match_trunk_qa(self, reaction_text: str, iteration: int):
        if self.trunks is None:
            return None

        trunk = self.current_trunks.get(iteration)
        if trunk is None or not trunk.question_answers:
            return None

        rows, qas = [], []
        for qa in trunk.question_answers:
            row = self.trunks.question_row(qa.qa_id)
            if row is not None:
                rows.append(row)
                qas.append(qa)
        if not rows:
            return None

        from ..trunks import embed as trunk_embed

        try:
            vector = trunk_embed.encode(reaction_text)
            hit = trunk_embed.best_match(vector, self.trunks.question_embeddings[rows])
        except Exception as err:
            if self.on_stage:
                self.on_stage(f"Trunk Q&A match unavailable ({type(err).__name__})")
            return None

        if hit is None:
            return None

        index, score = hit
        if score < config.TRUNK_QA_THRESHOLD:
            if self.on_stage and score > 0.0:
                self.on_stage(
                    f"No cached answer (best {score:.2f} < {config.TRUNK_QA_THRESHOLD:.2f}) "
                    f"— closest predicted: \"{qas[index].question[:60]}\""
                )
            return None
        return qas[index], score

    def _play_cached_answer(self, qa, score: float) -> bool:
        path = self.trunks.audio_path(qa.audio_file) if qa.has_audio else None
        self.publish("trunk_qa", qa.answer, str(path) if path else None)
        if self.on_stage:
            self.on_stage(
                f"Answered from the offline pool (similarity {score:.2f}, no LLM call)"
                + ("" if path else " — no audio for it, text only")
            )
        return path is not None

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
        qa_future = None

        if reaction_text.strip():
            qa_future = self._workers.submit(
                self._match_trunk_qa, reaction_text, turn.iteration,
            )

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

        if reaction_type == ReactionType.none and not requested:
            delta = config.ENGAGE_NONE

        anchor = "" if (reaction_type == ReactionType.none or requested) else anchor_snippet.strip()
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

        cached = None
        if qa_future is not None:
            try:
                cached = qa_future.result(timeout=config.WEB_FALLBACK_TIMEOUT_SECONDS)
            except Exception:
                cached = None

        if plan.needs_answer or cached is not None:
            if cached is not None:
                qa, score = cached
                reaction.answer = qa.answer
                reaction.answer_source = "trunk_qa"
                reaction.answer_match_score = round(score, 4)
                reaction.answer_qa_id = qa.qa_id
                self._play_cached_answer(qa, score)
            elif plan.answered:
                reaction.answer = plan.answer
                reaction.answer_source = "sources"
                if self.on_stage:
                    self.on_stage("Answered inline from the topic's sources (interpret)")
            else:
                with (timer.stage(timing.WEB_ANSWER) if timer else _null()):
                    reaction.answer, reaction.used_web = self._web_fallback(
                        plan, reaction.text, web_future,
                    )
                reaction.answer_source = "web" if reaction.used_web else (
                    "sources" if reaction.answer else "none"
                )
                if self.on_stage:
                    self.on_stage(
                        "Answered from live web retrieval"
                        if reaction.used_web
                        else f"No cached or web answer — fell back to plan.answer"
                    )

        if not reaction.answer_source and (
            reaction.type == ReactionType.question or _looks_like_question(reaction.text)
        ):
            reaction.answer_source = "unanswered"

        for pending in (web_future, qa_future):
            if pending is not None and not pending.done():
                pending.cancel()

        turn.reaction = reaction

        memory.apply_reaction(session_state.memory, reaction, delta=delta)

        turn.gist = plan.gist.strip() or summarize_turn(turn.text, self.llm)
        session_state.memory.summary = _update_summary(session_state.memory.summary, turn)

        memory.save_memory(session_state.memory)
        state.log_turn(session_state, turn.iteration)
        return turn


    def _write_answer_trace(self, session_state: SessionState) -> None:
        rows = []
        for turn in session_state.turns:
            reaction = turn.reaction
            if reaction is None or not reaction.text.strip():
                continue
            trunk = self.current_trunks.get(turn.iteration)
            rows.append({
                "iteration": turn.iteration,
                "topic": turn.topic,
                "trunk_id": getattr(trunk, "trunk_id", None),
                "asked": reaction.text,
                "reaction_type": reaction.type.value,
                "answer_source": reaction.answer_source or "(not a question)",
                "match_score": reaction.answer_match_score,
                "matched_qa_id": reaction.answer_qa_id or None,
                "answer": reaction.answer,
            })

        counts: dict[str, int] = {}
        for row in rows:
            counts[row["answer_source"]] = counts.get(row["answer_source"], 0) + 1

        payload = {
            "run_id": session_state.run_id,
            "trunk_pool_used": self.trunks is not None,
            "qa_threshold": config.TRUNK_QA_THRESHOLD,
            "counts": counts,
            "reactions": rows,
        }

        path = state.run_dir(session_state.run_id) / "answer_trace.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        if self.on_stage and counts:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            self.on_stage(f"Answer sources this session — {summary} (answer_trace.json)")

    def finish(self) -> SessionState:
    
        session_state = self._require_started()

        session_reactions = [t.reaction for t in session_state.turns if t.reaction]

        ### update persona style vector using llm call at the end of the session
        if not self._style_updated and session_reactions:
            self._style_updated = True
            try:
                memory.update_persona_style_from_session(
                    session_state.memory, self.persona, session_reactions, self.llm,
                )
            except Exception as err:
                if self.on_stage:
                    self.on_stage(f"persona style update failed: {type(err).__name__}: {err}")

        memory.save_memory(session_state.memory)
        transcript = "\n\n".join(f"[{t.topic}] {t.text}" for t in session_state.turns)
        (state.run_dir(session_state.run_id) / "session.txt").write_text(transcript)

        self._write_answer_trace(session_state)

        if self.timings:
            state.log_timings(session_state.run_id, self.timings)

        self._workers.shutdown(wait=False, cancel_futures=True)
        self._continuation.shutdown(wait=False, cancel_futures=True)

        return session_state

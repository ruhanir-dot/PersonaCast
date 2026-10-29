"""Does adapting upcoming predictions to the listener's in-session questions help? Simulated session.

The simulated listener plays a pool along its playback_sequence and asks its blind question
(evaluate_top_k cache) wherever it would. At each trunk t where it asks, after at least one
earlier question in the session, predictions are compared with its question at t under:
  static     the pipeline's offline predictions (as the player has them today)
  control    predict_actions now, with the persona's base history only (same predictor version
             as repredict, so the two differ only by the session questions)
  rerank     control's questions reordered by similarity to the session's earlier questions
             (layer 1: embedding only, no LLM)
  repredict  predict_actions with base history + the session's earlier questions as
             user_text (layer 2: background re-prediction)
Only questions asked BEFORE t enter its history, so nothing leaks from the future. Repredict
assumes the background call finishes before t starts playing. Questions in the session and the
test questions come from the same simulated listener, so a gain shows the predictor can pick up
this listener's way of asking within one session, not a real user's.

Example:
  python -m src.eval.session_adaptation --mind U76076 U9462 U23075 U1998 U52400 --jasmine
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.embedding_store import encode_texts
from src.eval.evaluate_top_k import VERSION, Cache, key_of, load_pool
from src.offline import predict_user_actions as pua
from src.utils import PROJECT_ROOT

EVAL = PROJECT_ROOT / "data" / "eval"
CONDITIONS = ("static", "control", "rerank", "repredict")
# Rerank score = similarity to the session's questions + PRIOR * (position bonus), so the
# predictor's own order still breaks near-ties (its first question is its best guess).
PRIOR = 0.02


def mind_persona(user):
    folder = EVAL / "mind" / "feeds" / f"mind_{user}"
    return {"name": f"mind_{user}", "user_id": f"mind_{user}", "profile": EVAL / "personas" / f"mind_{user}.json",
            "pool": folder / "output" / "trunk_pool_with_ranked_actions.json",
            "blind": folder / "eval" / "blind_questions.jsonl", "history": folder / "history.json"}


def jasmine_persona():
    return {"name": "jasmine", "user_id": "jasmine_v5", "profile": PROJECT_ROOT / "data/output/user_profile.json",
            "pool": PROJECT_ROOT / "data/output/trunk_pool_with_ranked_actions.json",
            "blind": EVAL / "results_v11" / "blind_questions.jsonl",
            "history": PROJECT_ROOT / "data/recent_runs/runs"}


def session_turn(number, key, node, question):
    return {"run_id": "current_session", "iteration": number, "current_trunk_id": key,
            "played_script": node["script"], "topic": node.get("title") or node.get("topic"),
            "user_text": question, "recorded_intent": "question", "recorded_type": "question"}


def rerank(questions, vectors, session):
    if not session:
        return list(questions)
    past = np.array([vectors[q] for q in session])
    scores = [float((past @ vectors[q]).max()) - PRIOR * i for i, q in enumerate(questions)]
    return [q for _, q in sorted(zip(scores, questions), key=lambda pair: -pair[0])]


def hits(asked, predicted, vectors, threshold):
    sims = [[float(vectors[asked[k]] @ vectors[q]) for q in predicted[k]] for k in asked]
    return {"hit1": round(float(np.mean([bool(s) and s[0] >= threshold for s in sims])), 3),
            "hit3": round(float(np.mean([bool(s) and max(s[:3]) >= threshold for s in sims])), 3)}


def run_persona(persona, args, store):
    raw_profile = pua.read(persona["profile"])
    profile = pua.profile_context(raw_profile)
    pool, nodes, static = load_pool(persona["pool"])
    blind = Cache(persona["blind"])
    route = [k for k in (pool.get("playback_sequence") or list(nodes)) if k in nodes]
    session = []  # (trunk, question) in playback order
    for key in route:
        answer = blind.rows.get(key_of(VERSION, "blind", profile, nodes[key]["script"]))
        if answer and answer["ask"]:
            session.append((key, answer["question"]))
    # Test on every asked trunk that has at least one earlier question in the session.
    tests = [(i, key, question) for i, (key, question) in enumerate(session) if i >= 1]
    try:
        base = pua.load_history(persona["history"], persona["user_id"], args.base_history,
                                include_interactions=False)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {persona['name']}: no base history ({exc})", flush=True)
        base = []
    print(f"{persona['name']}: {len(route)} trunks played, {len(session)} questions asked, "
          f"{len(tests)} test trunks, base history {len(base)}", flush=True)

    def histories(i):
        turns = [session_turn(n + 1, k, nodes[k], q) for n, (k, q) in enumerate(session[:i])]
        return {"control": base[-args.history_limit:],
                "repredict": (base + turns[-args.session_limit:])[-args.history_limit:]}

    def predict(job):
        name, i, key = job
        history = histories(i)[name]
        cache_key = key_of(pua.VERSION, persona["name"], name, history, nodes[key]["script"])
        if cache_key not in store.rows:
            try:
                questions = [a["question"] for a in pua.predict_actions(
                    profile, history, nodes[key], allow_not_interested=False)]
            except ValueError as exc:
                print(f"  {persona['name']} {name} {key} failed: {exc}", flush=True)
                questions = []
            store.put(cache_key, questions)
        return name, key, store.rows[cache_key]

    jobs = [(name, i, key) for name in ("control", "repredict") for i, key, _ in tests]
    predicted = {name: {} for name in CONDITIONS}
    with ThreadPoolExecutor(args.workers) as executor:
        for number, (name, key, questions) in enumerate(executor.map(predict, jobs), 1):
            predicted[name][key] = questions
            if number % 50 == 0:
                print(f"  [{number}/{len(jobs)}]", flush=True)
    asked = {key: question for _, key, question in tests}
    predicted["static"] = {key: static.get(key, []) for key in asked}

    texts = sorted({q for _, q in session} | {q for c in predicted.values() for qs in c.values() for q in qs})
    vectors = dict(zip(texts, encode_texts(texts)))
    for i, key, _ in tests:
        recent = [q for _, q in session[:i]][-args.session_limit:]
        predicted["rerank"][key] = rerank(predicted["control"][key], vectors, recent)

    result = {"played": len(route), "asked": len(session), "test_trunks": len(tests), "base_history": len(base),
              "conditions": {}}
    for name in CONDITIONS:
        row = {"questions_per_trunk": round(float(np.mean([len(predicted[name][k]) for k in asked])), 2)}
        for threshold in (0.75, 0.65):
            for metric, value in hits(asked, predicted[name], vectors, threshold).items():
                row[f"{metric}@{threshold}"] = value
        row["mean_best_cosine"] = round(float(np.mean([
            max([float(vectors[asked[k]] @ vectors[q]) for q in predicted[name][k]] or [0]) for k in asked])), 3)
        result["conditions"][name] = row
    # Does the gain grow as the session goes on? Split test trunks into first and second half.
    half = len(tests) // 2
    for part, chunk in (("early", tests[:half]), ("late", tests[half:])):
        subset = {key: asked[key] for _, key, _ in chunk}
        result[part] = {name: hits(subset, predicted[name], vectors, 0.75) for name in CONDITIONS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mind", nargs="*", default=[], help="MIND user ids, e.g. U76076")
    parser.add_argument("--jasmine", action="store_true", help="Also run Jasmine's own pool.")
    parser.add_argument("--history-limit", type=int, default=10, help="Total history the predictor sees.")
    parser.add_argument("--session-limit", type=int, default=5,
                        help="Most recent session questions used (repredict history and rerank).")
    parser.add_argument("--base-history", type=int, default=10, help="Persona history before the session.")
    parser.add_argument("--out-dir", type=Path, default=EVAL / "session_adaptation")
    parser.add_argument("--workers", type=int, default=int(os.getenv("LLM_WORKERS", "4")))
    args = parser.parse_args()
    personas = [mind_persona(u) for u in args.mind] + ([jasmine_persona()] if args.jasmine else [])
    if not personas:
        parser.error("Give --mind users and/or --jasmine.")
    store = Cache(args.out_dir / "predictions.jsonl")
    report = {"predictor": pua.VERSION, "history_limit": args.history_limit, "session_limit": args.session_limit,
              "rerank_prior": PRIOR, "personas": {}}
    for persona in personas:
        report["personas"][persona["name"]] = run_persona(persona, args, store)
        pua.save(args.out_dir / "summary.json", report)

    report["mean"] = {name: {m: round(float(np.mean([p["conditions"][name][m] for p in report["personas"].values()])), 3)
                             for m in ("hit1@0.75", "hit3@0.75", "hit1@0.65", "hit3@0.65", "mean_best_cosine")}
                      for name in CONDITIONS}
    pua.save(args.out_dir / "summary.json", report)
    print(f"\n{'persona':14} {'condition':10} {'Hit@1':>6} {'Hit@3':>6} {'H1@.65':>7} {'H3@.65':>7} {'best cos':>8}")
    rows = [(n, p["conditions"]) for n, p in report["personas"].items()] + [("MEAN", report["mean"])]
    for name, conditions in rows:
        for condition in CONDITIONS:
            r = conditions[condition]
            print(f"{name:14} {condition:10} {r['hit1@0.75']:>6.0%} {r['hit3@0.75']:>6.0%} {r['hit1@0.65']:>7.0%} "
                  f"{r['hit3@0.65']:>7.0%} {r['mean_best_cosine']:>8.3f}")
    print(f"Saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

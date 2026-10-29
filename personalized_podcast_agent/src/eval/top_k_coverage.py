"""Top-K coverage curve from the cached simulated-listener results (no LLM calls).

Run evaluate_top_k first with the same --persona arguments. For each pool, rater and K, the
global ranking (global_rank) is cut at K and compared with a random choice of K edges:
  coverage    share of trunks where the rater would ask whose blind question is matched
              (cosine >= threshold) by some question kept in the top K
  good_recall share of the pool's questions rated >= 4 that are kept in the top K
  precision   share of the kept questions rated >= 4

Example:
  python -m src.eval.top_k_coverage --k 50 100 150 \
      --persona jasmine=data/output/user_profile.json=data/output/trunk_pool_with_ranked_actions.json \
      --persona daniel=data/eval/personas/daniel.json=data/eval/daniel/ranked.json
"""
import argparse
import random
from pathlib import Path

import numpy as np

from src.embedding_store import encode_texts
from src.eval.evaluate_top_k import VERSION, Cache, key_of, load_pool
from src.offline.predict_user_actions import profile_context, read, save
from src.utils import PROJECT_ROOT


def metrics(kept, scores, blind, asked, good_total, matched):
    good = sum(scores.get(e, 0) >= 4 for e in kept)
    covered = {t for t in asked for e in kept if e[0] == t and matched[(t, e[1])]}
    return {"coverage": len(covered) / len(asked) if asked else None,
            "good_recall": good / good_total if good_total else None,
            "precision": good / len(kept) if kept else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona", action="append", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 150])
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "eval" / "results")
    parser.add_argument("--match-threshold", type=float, default=0.75)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    personas = {}
    for spec in args.persona:
        label, profile_path, pool_path = spec.split("=", 2)
        pool, nodes, predicted = load_pool(Path(pool_path))
        personas[label] = {"profile": profile_context(read(Path(profile_path))), "pool": pool,
                           "nodes": nodes, "predicted": predicted}
    scripts = {k: n["script"] for k, n in next(iter(personas.values()))["nodes"].items()}
    trunks = sorted(k for k in scripts if any(p["predicted"][k] for p in personas.values()))

    blind_cache = Cache(args.out_dir / "blind_questions.jsonl")
    rating_cache = Cache(args.out_dir / "ratings.jsonl")
    blind, scores = {}, {}
    for label, persona in personas.items():
        blind[label], scores[label] = {}, {}
        for trunk in trunks:
            value = blind_cache.rows.get(key_of(VERSION, "blind", persona["profile"], scripts[trunk]))
            if value and value["ask"]:
                blind[label][trunk] = value["question"]
            questions = sorted({q for p in personas.values() for q in p["predicted"][trunk]})
            random.Random(key_of(label, trunk)).shuffle(questions)
            value = rating_cache.rows.get(key_of(VERSION, "rate", persona["profile"], scripts[trunk], questions))
            for question, score in (value or {}).items():
                scores[label][(trunk, question)] = score
    missing = [l for l in personas if not blind[l]]
    if missing:
        raise SystemExit(f"No cached results for {missing}; run evaluate_top_k first.")

    texts = sorted({e["question"] for p in personas.values() for e in p["pool"]["ranked_actions"]} |
                   {q for b in blind.values() for q in b.values()})
    vectors = dict(zip(texts, encode_texts(texts)))

    report = {"version": VERSION, "match_threshold": args.match_threshold, "pairs": {}}
    rng = random.Random(args.seed)
    for pool_label, persona in personas.items():
        edges = sorted((e for e in persona["pool"]["ranked_actions"] if e["from_trunk_id"] in scripts),
                       key=lambda e: e["global_rank"])
        ranked = [(e["from_trunk_id"], e["question"]) for e in edges]
        for rater in personas:
            asked = set(blind[rater])
            matched = {e: e[0] in asked and float(vectors[blind[rater][e[0]]] @ vectors[e[1]])
                       >= args.match_threshold for e in ranked}
            rated = scores[rater]
            good_total = sum(rated.get(e, 0) >= 4 for e in ranked)
            rows = {}
            for k in sorted(set(args.k)) + [len(ranked)]:
                k = min(k, len(ranked))
                model = metrics(ranked[:k], rated, blind, asked, good_total, matched)
                trials = [metrics(rng.sample(ranked, k), rated, blind, asked, good_total, matched)
                          for _ in range(args.random_trials)]
                rows["all" if k == len(ranked) else str(k)] = {
                    "k": k, "model": model,
                    "random": {m: (float(np.mean([t[m] for t in trials])) if model[m] is not None else None)
                               for m in model}}
            report["pairs"][f"{pool_label}_pool__rated_by_{rater}"] = {
                "edges": len(ranked), "asked_trunks": len(asked), "good_questions": good_total,
                "trunk_ceiling": sum(any(matched[e] for e in ranked if e[0] == t) for t in asked),
                "by_k": rows}
    save(args.out_dir / "top_k_coverage.json", report)

    fmt = lambda v: "   -" if v is None else f"{v:4.0%}"
    for name, pair in report["pairs"].items():
        print(f"\n{name.replace('_pool__rated_by_', ' -> ')}: {pair['edges']} edges, "
              f"{pair['asked_trunks']} asked trunks, {pair['good_questions']} questions rated >=4")
        print(f"{'K':>5} | {'coverage':>15} | {'good recall':>15} | {'precision>=4':>15}")
        print(f"{'':>5} | {'model / random':>15} | {'model / random':>15} | {'model / random':>15}")
        for label, row in pair["by_k"].items():
            cells = [f"{fmt(row['model'][m])} / {fmt(row['random'][m])}"
                     for m in ("coverage", "good_recall", "precision")]
            print(f"{label:>5} | " + " | ".join(f"{c:>15}" for c in cells))
    print(f"\nSaved {args.out_dir / 'top_k_coverage.json'}")


if __name__ == "__main__":
    main()

"""Fetch the IG/YouTube feeds and news, merge daily data, and build personalized podcast answers.

The first step (fetch) opens Chrome with the saved login; log in once with
`python -m src.fetch_code.auto_fetch --login`. Use --from-step feeds to skip fetching.
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

STEPS = (
    ("fetch", "src.fetch_code.auto_fetch"),
    ("feeds", "src.user_profile.build_personal_feed"),
    ("trunks", "src.offline.generate_trunks"),
    ("pool", "src.offline.build_trunk_hierarchy"),
    ("actions", "src.offline.predict_user_actions"),
    ("rank", "src.offline.rank_user_actions"),
    ("answers", "src.offline.generate_answers"),
)
ADAPT_FIELDS = (
    "adapted_next_script", "adaptation_signature",
    "adaptation_model", "adaptation_prompt_version",
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def backup(path, folder):
    if path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, folder / path.name)


def supports_workers(source):
    """Inspect argparse declarations without importing models or running modules."""
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and any(isinstance(arg, ast.Constant) and arg.value == "--workers"
                for arg in node.args)
        for node in ast.walk(tree)
    )


def data_root(root, args):
    """Per-user data folder (<root>/daily, <root>/output); defaults to the project's data/."""
    folder = getattr(args, "data_root", None)
    if not folder:
        return root / "data"
    return folder if folder.is_absolute() else root / folder


def commands(root, args):
    base = data_root(root, args)
    output = base / "output"
    feeds = output / "personal_feed_items.json"
    trunks = output / "candidate_trunks.json"
    pool = output / "trunk_pool.json"
    predictions = output / "trunk_pool_predicted_actions.json"
    ranked = output / "trunk_pool_with_ranked_actions.json"
    final = output / "trunk_pool_with_actions.json"
    profile = args.profile or output / "user_profile.json"
    history = args.history or root / "data" / "recent_runs" / "runs"
    profile = profile if profile.is_absolute() else root / profile
    history = history if history.is_absolute() else root / history
    if not args.user_id:
        # The predictor rejects a profile whose user_id differs, so follow the profile by default.
        owner = load(profile).get("user_id") if profile.is_file() else None
        args.user_id = owner or "jasmine_v5"
    context = ["--user-id", args.user_id, "--profile", str(profile),
               "--history-limit", str(args.history_limit)]
    if args.history_limit:
        context += ["--history", str(history)]
    daily = base / "daily"
    fetching = args.from_step == "fetch"
    # --skip-browser: the manual lists are already in place (e.g. collected by search), but the
    # feeds must still be organized from them as after a browser fetch.
    skip_browser = fetching and getattr(args, "skip_browser", False)
    arguments = [
        [],
        # Freshly fetched feeds must be organized again; otherwise keep the prepared YouTube/IG.
        ["--from-step", "youtube" if fetching else "news"],
        ["--input", str(feeds), "--output", str(trunks)],
        ["--input", str(trunks), "--output", str(pool), "--checkpoint",
         str(output / "trunk_pool_build_checkpoint.json")],
        ["--input", str(pool), "--output", str(predictions), *context],
        ["--input", str(predictions), "--output", str(ranked), *context,
         "--top-k", str(args.top_k), "--min-score", str(args.min_score)],
        ["--input", str(ranked), "--output", str(final)],
    ]
    inputs = [root / "src" / "fetch_code" / "fetch_youtube.js",
              daily / ("youtube_homepage_manual.json" if fetching else "youtube_daily_items.json"),
              feeds, trunks, pool, predictions, ranked]
    outputs = [daily / "youtube_homepage_manual.json", feeds, trunks, pool, predictions, ranked, final]
    start = [name for name, _ in STEPS].index(args.from_step)
    plan = []
    for i, (name, module) in enumerate(STEPS):
        if i < start or (name == "fetch" and skip_browser):
            continue
        source = root.joinpath(*module.split(".")).with_suffix(".py")
        if not source.is_file():
            raise ValueError(f"Missing pipeline module: {source}")
        if args.fresh and name not in ("fetch", "feeds"):
            arguments[i].append("--no-resume" if name ==
                                "trunks" else "--fresh")
        if args.workers and name in ("trunks", "actions", "rank", "answers"):
            if supports_workers(source):
                arguments[i] += ["--workers", str(args.workers)]
            else:
                print(
                    f"{name}: module has no --workers option; using its normal execution.", flush=True)
        plan.append((name, [sys.executable, "-u", "-m", module, *arguments[i]],
                     inputs[i], outputs[i]))
    if not profile.is_file():
        raise ValueError(f"User profile missing: {profile}")
    if args.history_limit and not history.exists():
        raise ValueError(
            f"History missing: {history}. Use --history-limit 0 for cold start.")
    if args.from_step == "feeds":
        instagram = daily / "instagram_profile.json"
        # A user without Instagram has no manual IG file; the feed is then built without it.
        if not instagram.is_file() and (daily / "instagram_feed_manual.json").is_file():
            raise ValueError(f"Prepared Instagram data missing: {instagram}")
    return plan


def validate(path, step):
    if not path.is_file():
        raise ValueError(f"{step}: expected output missing: {path}")
    data = load(path)
    if step in ("fetch", "feeds", "trunks"):
        items = data if isinstance(data, list) else data.get(
            "feed_items" if step == "feeds" else "trunks")
        if not isinstance(items, list) or not items:
            raise ValueError(
                f"{step}: no items generated; check generation logs.")
        return
    if not isinstance(data, dict) or not data.get("nodes"):
        raise ValueError(f"{step}: output requires nonempty nodes.")
    nodes = data["nodes"]
    ids = {node["trunk_id"] for node in nodes}
    if len(ids) != len(nodes):
        raise ValueError(f"{step}: duplicate trunk IDs.")
    if step in ("actions", "rank", "answers"):
        if data.get("action_configuration", {}).get("complete") is not True:
            raise ValueError(
                "Action prediction incomplete; resume with --from-step actions.")
        for node in nodes:
            for action in node.get("action_pool", []):
                if action.get("action_type") != "continue" and action.get("next_trunk_id") not in ids:
                    raise ValueError(
                        f"{step}: invalid action destination in {node['trunk_id']}.")
    if step in ("rank", "answers"):
        cfg = data.get("ranking_configuration", {})
        if cfg.get("scoring_complete") is not True or cfg.get("selection_final") is not True:
            raise ValueError(
                "Action ranking incomplete; resume with --from-step rank.")
    if step == "answers":
        actions = [a for n in nodes for a in n.get("action_pool", [])
                   if a.get("action_type") != "continue"]
        answered = sum(bool(str(a.get("answer") or "").strip())
                       for a in actions)
        if answered != len(actions):
            raise ValueError(
                f"Answers incomplete: {answered}/{len(actions)}; resume with --from-step answers.")


def remove_adaptations(path, backup_dir):
    data = load(path)
    changed = "default_transitions" in data
    data.pop("default_transitions", None)
    for node in data.get("nodes", []):
        for action in node.get("action_pool", []):
            for field in ADAPT_FIELDS:
                if field in action:
                    changed = True
                    action.pop(field)
    if not changed:
        return
    backup(path, backup_dir)
    temporary = path.with_suffix(path.suffix + ".pipeline.tmp")
    temporary.write_text(json.dumps(
        data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.25 * (attempt + 1))


def main(argv=None, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-step", choices=[name for name, _ in STEPS], default="fetch")
    parser.add_argument("--user-id", help="Defaults to the profile's user_id (else jasmine_v5).")
    parser.add_argument("--data-root", type=Path,
                        help="Per-user data folder with daily/ and output/ (default: data/). "
                             "Its output/user_profile.json is the default profile.")
    parser.add_argument("--skip-browser", action="store_true",
                        help="With --from-step fetch: skip the browser fetch but still organize the "
                             "manual lists already in daily/.")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--history", type=Path)
    # 10: in the MIND context ablation, recent history lifted Hit@3 as much as the profile did.
    parser.add_argument("--history-limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--workers", type=int,
                        help="Forward only to modules that declare --workers.")
    parser.add_argument("--fresh", action="store_true",
                        help="Back up outputs and regenerate selected stages.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print commands only.")
    args = parser.parse_args(argv)
    if args.history_limit < 0 or args.top_k < 1 or not 1 <= args.min_score <= 5:
        parser.error(
            "Require history-limit >= 0, top-k >= 1, and min-score between 1 and 5.")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive.")
    root = Path(root).resolve() if root else Path(
        __file__).resolve().parents[2]
    base = data_root(root, args)
    backup_dir = base / "output" / "pipeline_backups" / \
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Child steps read their default locations from src.paths.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PODCAST_DATA_ROOT": str(base)}
    try:
        plan = commands(root, args)
        if not plan[0][2].is_file():
            raise ValueError(f"Required input missing: {plan[0][2]}")
        # Resume stages must consume complete upstream outputs.
        previous = {"trunks": "feeds", "pool": "trunks", "actions": "pool",
                    "rank": "actions", "answers": "rank"}
        if args.from_step in previous:
            validate(plan[0][2], previous[args.from_step])
        for name, command, _, _ in plan:
            print(f"{name}: {subprocess.list2cmdline(command)}", flush=True)
        if args.dry_run:
            return 0
        if args.fresh:
            paths = {row[3] for row in plan}
            if any(row[0] == "pool" for row in plan):
                paths.add(base / "output" / "trunk_pool_build_checkpoint.json")
            for path in sorted(paths):
                backup(path, backup_dir / "before_fresh")
        for number, (name, command, input_path, output_path) in enumerate(plan, 1):
            if not input_path.is_file():
                raise ValueError(f"Missing input for {name}: {input_path}")
            print(f"\n[{number}/{len(plan)}] Running {name}", flush=True)
            # UTF-8 output: titles may contain characters the Windows code page (cp950) lacks.
            subprocess.run(command, cwd=root, check=True, env=child_env)
            validate(output_path, name)
        final = plan[-1][3]
        remove_adaptations(final, backup_dir / "before_adaptation_cleanup")
        data = load(final)
        actions = [a for n in data["nodes"] for a in n.get("action_pool", [])
                   if a.get("action_type") != "continue"]
        selected = data["ranking_configuration"].get("selected_edges", 0)
        print(f"\nPipeline complete: {len(data['nodes'])} nodes, {len(actions)} answers, "
              f"{selected} globally selected edges (budget {args.top_k}).", flush=True)
        print("Ranking preserves all actions; answers are generated for every retained action.", flush=True)
        print(f"Saved to {final}", flush=True)
        return 0
    except subprocess.CalledProcessError as exc:
        print(
            f"\nStopped: child process exited with code {exc.returncode}; later stages were not run.", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 1
    except KeyboardInterrupt:
        print("\nStopped. Existing checkpoints retained.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        print(f"\nStopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

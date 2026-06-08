import argparse
import json
import os
import pickle
import time
from itertools import product

import pandas as pd

from config import ensure_dirs
from path_manager import PathManager

DEFAULT_CE_NAME = "CE_GTE_new"
DEFAULT_STAGE1_VER = "v5"
DEFAULT_CE_VER = "v5"
DEFAULT_JUDGE_VER = "v6"
DEFAULT_SPLIT = "test"
STAGE3_TOPK = 1000

TAU = 0.30
PENALTY = 0.85

ENSEMBLE_JUDGES = [
    "Judge_Qwen7B_old",
    "Judge_G4_31B_old",
    "Judge_G4_31B_new",
]

BLEND_GRID = [
    (0.25, 0.75),
    (0.30, 0.70),
    (0.35, 0.65),
]

CACHE_DIR = os.path.join(PathManager.BASE, "pipeline_cache")
ENS_DIR = os.path.join(PathManager.RESULTS_DIR, "ensemble_ablation")
SUBMISSIONS_DIR = os.path.join(ENS_DIR, "submissions")


def build_weight_grid() -> list[tuple[float, float, float]]:
    """
    All (qwen_w, g4old_w, g4new_w) triples where weights sum to 1.0, qwen_w is in [0.50, 0.95] in 0.05 steps,
    and Gemma budget is split between g4old and g4new in 0.05 steps.
    """
    triples = []
    for qw_int in range(50, 100, 5):
        qwen_w = round(qw_int / 100, 2)
        budget = round(1.0 - qwen_w, 2)
        for g4old_int in range(0, int(round(budget * 100)) + 1, 5):
            g4old_w = round(g4old_int / 100, 2)
            g4new_w = round(budget - g4old_w, 2)
            if g4new_w < -0.001:
                continue
            g4new_w = max(g4new_w, 0.0)
            triples.append((qwen_w, g4old_w, g4new_w))
    return triples


def build_blend_only_grid() -> list[tuple[float, float, float]]:
    """
    Minimal grid: just the best-known weight point with all blends.
    Fast sanity-check run (--blend-only).
    """
    return [(0.60, 0.20, 0.20)]


def make_run_id(ce_w: float, j_w: float, qwen_w: float, g4old_w: float, g4new_w: float) -> str:
    return (
        f"ENS_ABLATION_"
        f"ce{int(round(ce_w * 100))}j{int(round(j_w * 100))}_"
        f"t{int(round(TAU * 100))}_p{int(round(PENALTY * 100))}_"
        f"qw{int(round(qwen_w * 100))}_"
        f"go{int(round(g4old_w * 100))}_"
        f"gn{int(round(g4new_w * 100))}"
    )


def load_cache(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\n\tRun the pipeline with --cache flag first to generate '{label}' cache."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"\t{label}: {len(data)} entries OK  <- {os.path.basename(path)}")
    return data


def load_all_caches(args) -> tuple[list, list[str], dict, dict, list[str]]:
    """
    :return: Tuple containing:
                test_queries    : list of query dicts  {qid, query_text, ...}
                corpus_docids   : list[str]
                ce_results      : {qid_str: [(doc_idx, s2_score), ...]}
                judge_probs     : {adapter_name: {(qid_str, doc_idx): p_yes}}
                available       : adapters that were successfully loaded
    """
    print("\nLoading caches...")

    with open(PathManager.QUERIES_TEST_FILE, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    with open(PathManager.CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus_docids = [str(doc.get("docid", f"doc_{i}")) for i, doc in enumerate(corpus)]

    stage1_path = os.path.join(CACHE_DIR, f"stage1_{args.split}_ensemble_{args.stage1_ver}.pkl")
    load_cache(stage1_path, "Stage 1")

    ce_path = os.path.join(CACHE_DIR, f"ce_{args.ce_name}_{args.split}_ensemble_{args.ce_ver}.pkl")
    ce_results = load_cache(ce_path, f"CE ({args.ce_name})")

    n_expected = sum(len(ce_results[str(q["qid"])]) for q in test_queries)

    judge_probs: dict[str, dict] = {}
    for adapter in ENSEMBLE_JUDGES:
        jp_path = os.path.join(CACHE_DIR, f"judge_probs_{adapter}_{args.split}_{args.judge_ver}.pkl")
        if not os.path.exists(jp_path):
            print(f"\tWARNING: missing cache for {adapter} - skipping")
            continue
        jp = load_cache(jp_path, f"Judge ({adapter})")
        if len(jp) != n_expected:
            print(f"\tWARNING: size mismatch for {adapter} ({len(jp)} vs {n_expected} expected) - skipping")
            continue
        judge_probs[adapter] = jp

    if not judge_probs:
        raise RuntimeError(
            "No judge probability caches found.\n\tGenerate them by running pipeline.py with judge inference enabled."
        )

    available = [a for a in ENSEMBLE_JUDGES if a in judge_probs]
    print(f"\n\tLoaded {len(available)}/{len(ENSEMBLE_JUDGES)} judge caches: {available}")
    return test_queries, corpus_docids, ce_results, judge_probs, available


def build_ensemble_probs(
        qwen_w: float, g4old_w: float, g4new_w: float, judge_probs: dict, available: list[str]
) -> dict:
    """Weighted average of judge probabilities across available adapters."""
    raw_weights = {
        "Judge_Qwen7B_old": qwen_w,
        "Judge_G4_31B_old": g4old_w,
        "Judge_G4_31B_new": g4new_w,
    }
    weights = {a: w for a, w in raw_weights.items() if a in judge_probs and w > 0.0}
    wsum = sum(weights.values())
    if wsum == 0.0:
        raise ValueError(f"All ensemble weights are zero for available judges {available}.")

    ref_cache = judge_probs[available[0]]
    ens: dict = {}
    for key in ref_cache:
        num = den = 0.0
        for a, w in weights.items():
            p = judge_probs[a].get(key)
            if p is not None:
                num += w * p
                den += w
        ens[key] = num / den if den > 0.0 else 0.0
    return ens


def score_queries(
        ce_w: float, j_w: float,
        ens_probs: dict,
        test_queries: list, ce_results: dict, corpus_docids: list[str],
        run_id: str,
) -> list[dict]:
    """Apply ensemble probabilities + CE blend to produce ranked rows."""
    rows = []
    for query_obj in test_queries:
        qid = str(query_obj["qid"])
        ranked = ce_results[qid]

        scored = []
        for doc_idx, s2_score in ranked:
            yp = ens_probs[(qid, doc_idx)]
            final = ce_w * s2_score + j_w * yp
            if yp < TAU:
                final *= PENALTY
            scored.append((corpus_docids[doc_idx], final))

        scored.sort(key=lambda x: x[1], reverse=True)
        for rank, (docid, score) in enumerate(scored[:STAGE3_TOPK], 1):
            rows.append({
                "run_id": run_id,
                "manual": 0,
                "qid": qid,
                "docid": docid,
                "rank": rank,
                "score": round(score, 4),
            })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="JOKER ensemble weight ablation (CPU-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ce-name", default=DEFAULT_CE_NAME, help=f"CE model name (default: {DEFAULT_CE_NAME})")
    parser.add_argument(
        "--stage1-ver", default=DEFAULT_STAGE1_VER, help=f"Stage-1 cache version tag (default: {DEFAULT_STAGE1_VER})"
    )
    parser.add_argument("--ce-ver", default=DEFAULT_CE_VER, help=f"CE cache version tag (default: {DEFAULT_CE_VER})")
    parser.add_argument(
        "--judge-ver", default=DEFAULT_JUDGE_VER, help=f"Judge-prob cache version tag (default: {DEFAULT_JUDGE_VER})"
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, choices=["test", "dev", "train"], help="Query split to score (default: test)"
    )
    parser.add_argument(
        "--blend-only", action="store_true",
        help="Only run the best-known weight point with all blends (fast sanity check, ~3 combos)"
    )
    parser.add_argument("--max-combos", type=int, default=None, help="Cap total combos (for dev/debug)")
    parser.add_argument("--no-save", action="store_true", help="Score everything but do not write submission JSONs")
    return parser.parse_args()


def main():
    """
    Sweeps asymmetric judge-ensemble weights (Qwen, Gemma-old, Gemma-new) and CE/Judge blend ratios,
    generating one ranked submission per combo.

    This is a CPU-only experiment: all judge probabilities and CE scores must already be cached meaning produced
    by pipeline.py or a prior ablation run.

    Ablation axes
    -------------
      1. Qwen weight      : 0.50 -> 0.95 in 0.05 steps
      2. G4-old / G4-new  : asymmetric split of the remaining budget (0.05 steps)
      3. CE/Judge blend   : three fixed ratios (0.25/0.75, 0.30/0.70, 0.35/0.65)

    Fixed hyperparameters
      tau     = 0.30   (judge threshold below which a penalty is applied)
      penalty = 0.85   (score multiplier when judge prob < tau)

    Outputs
    -------
      results/ensemble_ablation/
        submissions/                              # one JSON per combo  (submission_<run_id>.json)
        ensemble_ablation_results.csv             # summary of all runs

    Cache files expected, produced by pipeline.py with --cache flag
      - pipeline_cache/stage1_<split>_ensemble_<ver>.pkl
      - pipeline_cache/ce_<CE_NAME>_<split>_ensemble_<ver>.pkl
      - pipeline_cache/judge_probs_<adapter>_<split>_<ver>.pkl

    Usage
    -----
      python ensemble_ablation.py                          # defaults
      python ensemble_ablation.py --split test             # official test queries
      python ensemble_ablation.py --blend-only             # skip weight sweep, blends only
      python ensemble_ablation.py --max-combos 100         # cap total combos (dev/debug)
      python ensemble_ablation.py --no-save                # score only, no JSON files
      python ensemble_ablation.py --stage1-ver v5 --ce-ver v5 --judge-ver v6
    """
    args = parse_args()
    ensure_dirs()
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    test_queries, corpus_docids, ce_results, judge_probs, available = \
        load_all_caches(args)

    weight_grid = build_blend_only_grid() if args.blend_only else build_weight_grid()
    all_combos = list(product(weight_grid, BLEND_GRID))
    if args.max_combos:
        all_combos = all_combos[:args.max_combos]

    total = len(all_combos)
    print(f"\n{'-' * 60}")
    print(f"\tEnsemble weight ablation")
    print(f"\tWeight triples : {len(weight_grid)}")
    print(f"\tCE/J blends    : {len(BLEND_GRID)}")
    print(f"\tTotal combos   : {total}")
    print(f"\tSave JSONs     : {not args.no_save}")
    print(f"{'-' * 60}\n")

    results_csv = os.path.join(ENS_DIR, "ensemble_ablation_results.csv")
    done_runs: set[str] = set()
    all_results: list[dict] = []

    if os.path.exists(results_csv):
        df_prev = pd.read_csv(results_csv)
        done_runs = set(df_prev["run_id"].tolist())
        all_results = df_prev.to_dict("records")
        print(f"\tResuming: {len(done_runs)} runs already in CSV, skipping.\n")

    existing_files = set()
    if not args.no_save:
        existing_files = {f for f in os.listdir(SUBMISSIONS_DIR) if f.startswith("submission_ENS_ABLATION")}

    done = skipped = saved = 0
    t0 = time.time()

    for (qwen_w, g4old_w, g4new_w), (ce_w, j_w) in all_combos:
        run_id = make_run_id(ce_w, j_w, qwen_w, g4old_w, g4new_w)
        done += 1

        if run_id in done_runs:
            skipped += 1
            continue

        fname = f"submission_{run_id}.json"
        if fname in existing_files:
            skipped += 1
            continue

        t_run = time.time()
        ens_probs = build_ensemble_probs(
            qwen_w, g4old_w, g4new_w, judge_probs, available
        )
        rows = score_queries(
            ce_w, j_w, ens_probs,
            test_queries, ce_results, corpus_docids, run_id,
        )
        elapsed = time.time() - t_run

        out_path = None
        if not args.no_save:
            out_path = os.path.join(SUBMISSIONS_DIR, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, separators=(",", ":"))
            saved += 1

        result = {
            "run_id": run_id,
            "ce_name": args.ce_name,
            "ce_w": ce_w,
            "j_w": j_w,
            "qwen_w": qwen_w,
            "g4old_w": g4old_w,
            "g4new_w": g4new_w,
            "tau": TAU,
            "penalty": PENALTY,
            "n_rows": len(rows),
            "time_sec": round(elapsed, 2),
            "file": fname if out_path else "",
        }
        all_results.append(result)
        del rows

        new_total = saved + (done - skipped - saved)
        if (saved > 0 and saved % 20 == 0) or done == total:
            pd.DataFrame(all_results).to_csv(results_csv, index=False)
            elapsed_total = time.time() - t0
            eta = elapsed_total / max(saved, 1) * max(total - done, 0)
            size_str = ""
            if out_path and os.path.exists(out_path):
                size_mb = os.path.getsize(out_path) / 1_048_576
                size_str = f"\t({size_mb:.0f} MB/file)"
            print(f"\t{done}/{total}  saved={saved}  skipped={skipped}{size_str}  ETA: {eta:.0f}s")

    if all_results:
        pd.DataFrame(all_results).to_csv(results_csv, index=False)

    total_elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE ABLATION COMPLETE in {total_elapsed:.0f}s")
    print(f"\tTotal combos   : {total}")
    print(f"\tNew saves      : {saved}")
    print(f"\tSkipped        : {skipped}")
    if not args.no_save:
        all_files = [f for f in os.listdir(SUBMISSIONS_DIR) if f.startswith("submission_ENS_ABLATION")]
        total_gb = sum(os.path.getsize(os.path.join(SUBMISSIONS_DIR, f)) for f in all_files) / 1_073_741_824
        print(f"\tFiles on disk  : {len(all_files)}  ({total_gb:.2f} GB)")
        print(f"\tFolder         : {SUBMISSIONS_DIR}")
    print(f"\tCSV summary    : {results_csv}")
    print(f"{'=' * 60}")
    print()

    if all_results:
        df = pd.DataFrame(all_results)
        print("Config table (sorted by run_id, no MAP yet - evaluate on CodaBench):")
        print(df[["run_id", "ce_w", "j_w", "qwen_w", "g4old_w", "g4new_w", "n_rows"]]
              .sort_values("run_id")
              .to_string(index=False))
        print()
        print("To submit the best file:")
        print("\tcp <SUBMISSIONS_DIR>/submission_<run_id>.json prediction.json")
        print("\tzip prediction.zip prediction.json")
        print("\t# Upload prediction.zip to CodaBench")


if __name__ == "__main__":
    main()

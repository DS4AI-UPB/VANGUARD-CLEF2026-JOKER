import argparse
import os
import time
from collections import defaultdict

import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from iroh.core.config import CE_CANDIDATES, JUDGE_CANDIDATES, JUDGE_BASE_MODEL, USE_EXPANDED_BM25, ensure_dirs
from iroh.core.path_manager import PathManager
from iroh.core.pipeline import (
    build_bm25_index, load_dense_embeddings, run_pipeline, precompute_stage1, CorpusData, JudgeData
)
from iroh.core.utils import (
    seed_everything, load_json,
    load_judge_model, evaluate_trec, free_gpu,
    load_corpus, build_qrel_dict, load_humor_prior,
    stage_recall_record,
)


def discover_runs(max_judges: int = 2, full: bool = False) -> list[dict]:
    """
    Build experiment grid from available models on disk.

    Returns runs for:
      1. Every CE with Judge combination (capped by max_judges unless --full)
      2. Every CE with no judge (CE-only ablation)
      3. Every judge with no CE (Judge-only ablation)
      4. Baseline: no CE, no judge
    """
    available_ces = [name for name in CE_CANDIDATES if os.path.exists(os.path.join(PathManager.MODELS_DIR, name))]
    available_judges = [name for name in JUDGE_CANDIDATES if os.path.exists(os.path.join(PathManager.MODELS_DIR, name))]

    print(f"Available CEs:     {available_ces}")
    print(f"Available Judges:  {available_judges}")

    runs = []
    judge_limit = len(available_judges) if full else max_judges

    for ce_name in available_ces:
        for judge_name in available_judges[:judge_limit]:
            runs.append({
                "name": f"{ce_name}__{judge_name}",
                "ce_path": os.path.join(PathManager.MODELS_DIR, ce_name),
                "judge_path": os.path.join(PathManager.MODELS_DIR, judge_name),
                "ablation_type": "CE+Judge",
            })

    for ce_name in available_ces:
        runs.append({
            "name": f"{ce_name}__NoJudge",
            "ce_path": os.path.join(PathManager.MODELS_DIR, ce_name),
            "judge_path": None,
            "ablation_type": "CE_only",
        })

    for judge_name in available_judges[:judge_limit]:
        runs.append({
            "name": f"NoCE__{judge_name}",
            "ce_path": None,
            "judge_path": os.path.join(PathManager.MODELS_DIR, judge_name),
            "ablation_type": "Judge_only",
        })

    runs.append({
        "name": "Baseline__NoCE_NoJudge",
        "ce_path": None,
        "judge_path": None,
        "ablation_type": "Baseline",
    })

    return runs


def main():
    """
    Run the pipeline across all component combinations to measure each component's contribution.

    Ablation dimensions:
      - CE with Judge  (full pipeline, all combos)
      - CE only        (no judge - measures CE contribution)
      - Judge only     (no CE - measures judge contribution)
      - Baseline       (no CE, no judge - BM25 + dense + humor prior only)

    Usage:
        python ablation.py                        # test all combos
        python ablation.py --max-judges 3         # limit judges per CE
        python ablation.py --full                 # no limits, all with all
    """
    parser = argparse.ArgumentParser(description="JOKER ablation study")
    parser.add_argument("--max-judges", type=int, default=2, help="Max judge models to pair with each CE")
    parser.add_argument("--full", action="store_true", help="Test ALL CE with ALL Judge combinations (no limits)")
    args = parser.parse_args()

    seed_everything()
    ensure_dirs()

    results_dir = os.path.join(PathManager.RESULTS_DIR, "ablation")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading shared resources...")
    corpus, corpus_texts, corpus_docids = load_corpus(PathManager.CORPUS_FILE)

    test_queries = load_json(PathManager.LOCAL_TEST_QUERIES)
    qrels_data = load_json(PathManager.LOCAL_TEST_QRELS)
    qrel_dict = build_qrel_dict(qrels_data)

    humor_prior, humor_prior_array = load_humor_prior(corpus_docids)

    bm25 = build_bm25_index(corpus, corpus_texts, corpus_docids, USE_EXPANDED_BM25)
    embedder, corpus_embeddings = load_dense_embeddings(corpus_texts)

    corpus_data = CorpusData(
        corpus_texts=corpus_texts,
        corpus_docids=corpus_docids,
        bm25=bm25,
        embedder=embedder,
        corpus_embeddings=corpus_embeddings,
        humor_prior=humor_prior,
        humor_prior_array=humor_prior_array,
    )

    runs = discover_runs(max_judges=args.max_judges, full=args.full)

    type_counts = defaultdict(int)
    for r in runs:
        type_counts[r["ablation_type"]] += 1
    print(f"\n{len(runs)} configurations to test:")
    for atype, count in type_counts.items():
        print(f"\t{atype}: {count}")

    done_csv = os.path.join(results_dir, "ablation_results.csv")
    done_runs = set()
    all_results = []
    if os.path.exists(done_csv):
        df = pd.read_csv(done_csv)
        done_runs = set(df["run_name"].tolist())
        all_results = df.to_dict("records")
        print(f"\n  {len(done_runs)} already completed, skipping")

    runs = [r for r in runs if r["name"] not in done_runs]

    if not runs:
        print("All runs already done!")
    else:
        print("\nPre-computing Stage 1 (computed once, reused for every run)...")
        stage1_cache = precompute_stage1(test_queries, corpus_data)
        import gc
        del corpus_data.embedder
        gc.collect()
        torch.cuda.empty_cache()

        ce_stage2_caches: dict[str, dict] = {}

        runs_by_judge = defaultdict(list)
        for run in runs:
            key = run["judge_path"] or "none"
            runs_by_judge[key].append(run)

        for judge_key, judge_runs in runs_by_judge.items():
            judge = JudgeData()

            if judge_key != "none":
                print(f"\nLoading judge: {os.path.basename(judge_key)}...")
                judge_model, judge_tokenizer, yes_id, no_id = load_judge_model(judge_key, JUDGE_BASE_MODEL)
                judge = JudgeData(model=judge_model, tokenizer=judge_tokenizer, yes_id=yes_id, no_id=no_id)

            for run in judge_runs:
                print(f"\n{'-' * 60}")
                print(f"\tRUN: {run['name']}  [{run['ablation_type']}]")
                print(f"{'-' * 60}")

                cross_encoder = None
                ce_key = run["ce_path"] or "none"
                if run["ce_path"] and os.path.exists(run["ce_path"]):
                    cross_encoder = CrossEncoder(run["ce_path"])

                if ce_key not in ce_stage2_caches:
                    ce_stage2_caches[ce_key] = {}
                    if cross_encoder is not None:
                        print(f"\tStage-2 cache: new (first run with CE '{os.path.basename(ce_key)}')")
                else:
                    print(f"\tStage-2 cache: reusing {len(ce_stage2_caches[ce_key])} "
                          f"cached queries for CE '{os.path.basename(ce_key)}'")

                start_time = time.time()
                submission_data, stage_recalls = run_pipeline(
                    run_name=run["name"],
                    test_queries=test_queries,
                    qrel_dict=qrel_dict,
                    corpus_data=corpus_data,
                    cross_encoder=cross_encoder,
                    judge=judge,
                    stage1_cache=stage1_cache,
                    ce_stage2_cache=ce_stage2_caches[ce_key],
                )
                elapsed = time.time() - start_time

                eval_metrics = evaluate_trec(submission_data, qrels_data)

                result = {
                    "run_name": run["name"],
                    "ablation_type": run["ablation_type"],
                    "ce": os.path.basename(run["ce_path"]) if run["ce_path"] else "none",
                    "judge": os.path.basename(run["judge_path"]) if run["judge_path"] else "none",
                    "MAP": round(eval_metrics.get("map", 0), 4),
                    "R@30": round(eval_metrics.get("recall_30", 0), 4),
                    "R@1000": round(eval_metrics.get("recall_1000", 0), 4),
                    "NDCG@10": round(eval_metrics.get("ndcg_cut_10", 0), 4),
                    "P@5": round(eval_metrics.get("P_5", 0), 4),
                    "P@10": round(eval_metrics.get("P_10", 0), 4),
                    **stage_recall_record(stage_recalls),
                    "time_sec": round(elapsed, 1),
                }
                all_results.append(result)

                print(f"\tMAP={result['MAP']:.4f}  NDCG@10={result['NDCG@10']:.4f}  "
                      f"R@30={result['R@30']:.4f}  R@1000={result['R@1000']:.4f}")

                pd.DataFrame(all_results).to_csv(done_csv, index=False)

                if cross_encoder is not None:
                    del cross_encoder
                    torch.cuda.empty_cache()

            if judge.is_loaded:
                free_gpu(judge.model)

    if all_results:
        best = max(all_results, key=lambda x: x["MAP"])
        print(f"\n{'=' * 70}")
        print(f"ABLATION COMPLETE - {len(all_results)} configurations tested")
        print(f"{'=' * 70}")
        print(f"\nBest overall: {best['run_name']}")
        print(f"\tMAP={best['MAP']:.4f}  NDCG@10={best['NDCG@10']:.4f}  R@30={best['R@30']:.4f}")

        by_type = defaultdict(list)
        for r in all_results:
            by_type[r.get("ablation_type", "unknown")].append(r)

        for atype in ["Baseline", "CE_only", "Judge_only", "CE+Judge"]:
            group = by_type.get(atype, [])
            if not group:
                continue
            group_best = max(group, key=lambda x: x["MAP"])
            print(f"\n-- {atype} ({len(group)} runs) --")
            print(f"\tBest: {group_best['run_name']}")
            print(f"\tMAP={group_best['MAP']:.4f}  NDCG@10={group_best['NDCG@10']:.4f}")

        baseline_map = 0.0
        baseline_runs = by_type.get("Baseline", [])
        if baseline_runs:
            baseline_map = baseline_runs[0]["MAP"]

        ce_only = by_type.get("CE_only", [])
        judge_only = by_type.get("Judge_only", [])
        full_runs = by_type.get("CE+Judge", [])

        if baseline_map > 0 and (ce_only or judge_only or full_runs):
            print(f"\n-- Component Contribution (vs baseline MAP={baseline_map:.4f}) --")
            if ce_only:
                best_ce = max(ce_only, key=lambda x: x["MAP"])
                print(f"\t+ CE alone:        MAP={best_ce['MAP']:.4f}  "
                      f"(Δ={best_ce['MAP'] - baseline_map:+.4f})")
            if judge_only:
                best_j = max(judge_only, key=lambda x: x["MAP"])
                print(f"\t+ Judge alone:     MAP={best_j['MAP']:.4f}  "
                      f"(Δ={best_j['MAP'] - baseline_map:+.4f})")
            if full_runs:
                best_full = max(full_runs, key=lambda x: x["MAP"])
                print(f"\t+ CE + Judge:      MAP={best_full['MAP']:.4f}  "
                      f"(Δ={best_full['MAP'] - baseline_map:+.4f})")

        print(f"\n{'Run':<55} {'Type':<12} {'MAP':>6} {'NDCG':>6} {'R@30':>6} {'R@1K':>6}")
        print("-" * 90)
        for m in sorted(all_results, key=lambda x: x["MAP"], reverse=True):
            print(f"{m['run_name']:<55} {m.get('ablation_type', ''):<12} "
                  f"{m['MAP']:>6.4f} {m['NDCG@10']:>6.4f} "
                  f"{m['R@30']:>6.4f} {m['R@1000']:>6.4f}")


if __name__ == "__main__":
    main()

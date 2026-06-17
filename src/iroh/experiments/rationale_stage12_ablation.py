import argparse
import os
import time
import zipfile

import pandas as pd
from sentence_transformers import CrossEncoder

from iroh.core.config import CE_GTE_BASE_MODEL, ensure_dirs
from iroh.core.path_manager import PathManager
from iroh.core.pipeline import build_bm25_index, load_dense_embeddings, precompute_stage1, run_pipeline, CorpusData, JudgeData
from iroh.core.utils import seed_everything, load_json, save_json, load_corpus, build_qrel_dict, load_humor_prior, evaluate_trec


def _load_ce(name: str | None):
    """Load a trained cross-encoder by name under MODELS_DIR, or None."""
    if name is None:
        return None
    path = os.path.join(PathManager.MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cross-encoder '{name}' not found at {path}. "
            f"Train it first (train_cross_encoder.py) or pass --train-base-ce."
        )
    return CrossEncoder(path)


def _eval_cell(run_name, corpus_data, ce, test_queries, qrel_dict,
               qrels_data, stage1_cache):
    """Run Stages 1-2 (judge OFF) for one cell and return (submission, metrics)."""
    judge = JudgeData()
    submission, _recalls = run_pipeline(
        run_name=run_name,
        test_queries=test_queries,
        qrel_dict=qrel_dict,
        corpus_data=corpus_data,
        cross_encoder=ce,
        judge=judge,
        stage1_cache=stage1_cache,
        ce_stage2_cache={},  # fresh per cell: Stage 1 differs by BM25 mode
    )
    metrics = evaluate_trec(submission, qrels_data) if qrels_data else {}
    return submission, metrics


def _row(run_name, bm25_mode, ce_label, metrics):
    return {
        "run_name": run_name,
        "bm25": bm25_mode,
        "ce": ce_label,
        "MAP": round(metrics.get("map", 0.0), 4),
        "R@30": round(metrics.get("recall_30", 0.0), 4),
        "R@1000": round(metrics.get("recall_1000", 0.0), 4),
        "NDCG@10": round(metrics.get("ndcg_cut_10", 0.0), 4),
        "P@10": round(metrics.get("P_10", 0.0), 4),
    }


def _maybe_train_base_ce(name: str):
    """
    OPTIONAL, HEAVY. Train a cross-encoder on the no-rationale base data (processed_joker_train.json)
    so the CE axis has a true non-distilled point.

    Defaults to the GTE backbone to match the paper primary CE.
    """
    from iroh.train.train_cross_encoder import train_one_config

    base_model_id = CE_GTE_BASE_MODEL
    is_gte = "gte" in base_model_id.lower()
    cfg = {
        "name": name,
        "rationale_file": PathManager.PROCESSED_TRAIN_FILE,
        "augmented_file": None,
        "use_augmented": False,
        "base_model": base_model_id,
        "batch_size": 8 if is_gte else 128,
        "lr": 2e-5 if is_gte else 1e-5,
        "max_epochs": 50,
        "patience": 3,
        "warmup_ratio": 0.15,
        "weight_decay": 0.02,
        "automodel_args": ({"torch_dtype": "auto", "attn_implementation": "eager"} if is_gte else None),
    }
    corpus = load_json(PathManager.CORPUS_FILE)
    corpus_texts = [d.get("text", "") for d in corpus]
    print(f"\nTraining no-rationale base CE '{name}' on {os.path.basename(PathManager.PROCESSED_TRAIN_FILE)} ...")
    train_one_config(cfg, corpus_texts)


def _write_submission(submission, run_name, out_dir):
    """Save a CodaBench-style submission JSON and a matching .zip."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{run_name}.json")
    save_json(submission, json_path)
    zip_path = os.path.join(out_dir, f"{run_name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, f"{run_name}.json")
    print(f"\tSubmission: {json_path}")
    print(f"\tZipped:     {zip_path}  (upload this to CodaBench)")


def main():
    """
    Isolates how much the generated rationales contribute to the retrieval stages, with the Stage-3 judge switched off.
    This is the experiment that backs the "rationale-distilled vs. non-distilled" claim at the retrieval level.

    Output (local test split, which has qrels):
        - Stage 1 only: plain BM25 + BGE
        - Stage 1 only: rationale-expanded BM25 + BGE
        - Stage 1 + 2:  plain BM25 + BGE + CE
        - Stage 1 + 2:  rationale-expanded BM25 + BGE + CE
    with MAP / NDCG@10 / recall and the rationale delta at each depth.

    Usage:
        python rationale_stage12_ablation.py --ce CE_GTE_new
        python rationale_stage12_ablation.py --ce CE_GTE_new --ce-base CE_GTE_base
        python rationale_stage12_ablation.py --ce CE_GTE_new --submission
        python rationale_stage12_ablation.py --train-base-ce CE_GTE_base   # optional, heavy
    """
    parser = argparse.ArgumentParser(description="Rationale ablation for Stages 1-2 (judge OFF).")
    parser.add_argument(
        "--ce", type=str, default=None,
        help="Cross-encoder name under MODELS_DIR (rationale-trained). Omit to run Stage-1-only comparison."
    )
    parser.add_argument(
        "--ce-base", type=str, default=None,
        help="Optional second CE name (e.g. trained on no-rationale base data) to add a CE-axis comparison."
    )
    parser.add_argument(
        "--train-base-ce", type=str, default=None,
        help="OPTIONAL/HEAVY: train a no-rationale CE with this name on processed_joker_train.json, then exit."
    )
    parser.add_argument(
        "--submission", action="store_true",
        help="Also emit CodaBench submission files for the two endpoint configs (plain+CE vs rationale+CE)."
    )
    args = parser.parse_args()

    seed_everything()
    ensure_dirs()

    if args.train_base_ce:
        _maybe_train_base_ce(args.train_base_ce)
        return

    out_dir = os.path.join(PathManager.RESULTS_DIR, "rationale_ablation")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading corpus + BGE dense embeddings (shared across all cells)...")
    corpus, corpus_texts, corpus_docids = load_corpus(PathManager.CORPUS_FILE)
    humor_prior, humor_prior_array = load_humor_prior(corpus_docids)
    embedder, corpus_embeddings = load_dense_embeddings(corpus_texts)

    if not os.path.exists(PathManager.RATIONALES_FILE):
        print(
            f"WARNING: {PathManager.RATIONALES_FILE} not found - the rationale-expanded BM25 "
            f"will fall back to plain text and both rows will match."
        )

    print("\nBuilding BM25 indexes (plain vs rationale-expanded)...")
    bm25_plain = build_bm25_index(corpus, corpus_texts, corpus_docids, use_expanded=False)
    bm25_rat = build_bm25_index(corpus, corpus_texts, corpus_docids, use_expanded=True)

    def make_corpus_data(bm25):
        return CorpusData(
            corpus_texts=corpus_texts,
            corpus_docids=corpus_docids,
            bm25=bm25,
            embedder=embedder,
            corpus_embeddings=corpus_embeddings,
            humor_prior=humor_prior,
            humor_prior_array=humor_prior_array,
        )

    cd_plain = make_corpus_data(bm25_plain)
    cd_rat = make_corpus_data(bm25_rat)

    test_queries = load_json(PathManager.LOCAL_TEST_QUERIES)
    qrels_data = load_json(PathManager.LOCAL_TEST_QRELS)
    qrel_dict = build_qrel_dict(qrels_data)
    print(f"\nLocal eval: {len(test_queries)} queries, {len(qrels_data)} qrels")

    print("\nPre-computing Stage 1 (plain BM25)...")
    s1_plain = precompute_stage1(test_queries, cd_plain)
    print("Pre-computing Stage 1 (rationale BM25)...")
    s1_rat = precompute_stage1(test_queries, cd_rat)

    ce_specs = [("none", None)]
    if args.ce:
        ce_specs.append((args.ce, _load_ce(args.ce)))
    if args.ce_base:
        ce_specs.append((args.ce_base, _load_ce(args.ce_base)))

    rows = []
    for bm25_mode, cd, s1 in [("plain", cd_plain, s1_plain),
                              ("rationale", cd_rat, s1_rat)]:
        for ce_label, ce in ce_specs:
            run_name = f"abl_{bm25_mode}_BM25__CE_{ce_label}"
            print(f"\n{'-' * 60}\n  {run_name}\n{'-' * 60}")
            t0 = time.time()
            _sub, metrics = _eval_cell(
                run_name, cd, ce, test_queries, qrel_dict, qrels_data, s1
            )
            row = _row(run_name, bm25_mode, ce_label, metrics)
            row["time_sec"] = round(time.time() - t0, 1)
            rows.append(row)
            print(
                f"\tMAP={row['MAP']:.4f}  NDCG@10={row['NDCG@10']:.4f}  "
                f"R@30={row['R@30']:.4f}  R@1000={row['R@1000']:.4f}"
            )
            if ce is not None:
                del ce
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "rationale_ablation_local.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}\nRATIONALE ABLATION - local test split (judge OFF)\n{'=' * 70}")
    print(f"{'BM25':<11}{'CE':<16}{'MAP':>8}{'NDCG@10':>10}{'R@30':>8}{'R@1000':>9}")
    print("-" * 62)
    for r in rows:
        print(
            f"{r['bm25']:<11}{r['ce']:<16}{r['MAP']:>8.4f}{r['NDCG@10']:>10.4f}{r['R@30']:>8.4f}{r['R@1000']:>9.4f}"
        )

    print("\nRationale delta (rationale BM25 - plain BM25), same CE:")
    by_ce = {}
    for r in rows:
        by_ce.setdefault(r["ce"], {})[r["bm25"]] = r
    for ce_label, d in by_ce.items():
        if "plain" in d and "rationale" in d:
            dmap = d["rationale"]["MAP"] - d["plain"]["MAP"]
            dndcg = d["rationale"]["NDCG@10"] - d["plain"]["NDCG@10"]
            print(f"\tCE={ce_label:<16}  ΔMAP={dmap:+.4f}  ΔNDCG@10={dndcg:+.4f}")

    print(f"\nSaved: {csv_path}")

    if args.submission:
        if not args.ce:
            print("\n--submission needs --ce (the rationale-trained CE). Skipping.")
            return
        print(f"\n{'=' * 70}\nGenerating CodaBench submissions (official test queries)\n{'=' * 70}")
        official_queries = load_json(PathManager.QUERIES_TEST_FILE)
        sub_dir = os.path.join(out_dir, "submissions")

        print("\nPre-computing Stage 1 on official queries...")
        s1_plain_off = precompute_stage1(official_queries, cd_plain)
        s1_rat_off = precompute_stage1(official_queries, cd_rat)

        for bm25_mode, cd, s1off in [("plain", cd_plain, s1_plain_off), ("rationale", cd_rat, s1_rat_off)]:
            ce = _load_ce(args.ce)
            run_name = f"VANGUARD_Task1_abl_{bm25_mode}_BM25__CE_{args.ce}__NoJudge"
            submission, _ = run_pipeline(
                run_name=run_name,
                test_queries=official_queries,
                qrel_dict={},
                corpus_data=cd,
                cross_encoder=ce,
                judge=JudgeData(),
                stage1_cache=s1off,
                ce_stage2_cache={},
            )
            _write_submission(submission, run_name, sub_dir)
            del ce


if __name__ == "__main__":
    main()

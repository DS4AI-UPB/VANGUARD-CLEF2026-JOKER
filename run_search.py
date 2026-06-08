import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import itertools
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from config import CE_CANDIDATES, USE_EXPANDED_BM25, ensure_dirs, EMBEDDER_CANDIDATES
from path_manager import PathManager
from utils import (
    build_qrel_dict, evaluate_trec, expand_query,
    load_corpus, load_humor_prior,
    load_json, save_json,
    seed_everything, stage_recall_record, tokenize_stemmed,
)
from pipeline import CorpusData, _rrf_fuse, _dedup_candidates, load_dense_embeddings

S1_GRID = {
    "stage1_topk": [1000, 2000, 3000],
    "bm25_k1": [1.0, 1.2, 1.5],
    "bm25_b": [0.5, 0.75],
    "rrf_k": [30, 60],
    "rrf_bm25_weight": [0.8, 1.0, 1.2],
    "query_expansion_variants": [2, 3, 5],
}

S2_GRID = {
    "stage2_topk": [300, 500, 700],
    "stage2_ce_blend": [0.55, 0.70, 0.80, 0.90],
    "stage2_dedup": [0.93, 0.95, 0.97, 1.00],
}

S2_DEFAULTS = dict(stage2_topk=500, stage2_ce_blend=0.70, stage2_dedup=0.97)

KNOWN_GOOD_CONFIG = {
    "stage1_topk": 2000,
    "bm25_k1": 1.2,
    "bm25_b": 0.5,
    "rrf_k": 60,
    "rrf_bm25_weight": 1.0,
    "query_expansion_variants": 3,
    "stage2_topk": 500,
    "stage2_ce_blend": 0.85,
    "stage2_dedup": 0.97,
}

CE_PRIORITY = CE_CANDIDATES

DEFAULT_TOP_N = 10
BEST_CONFIG_FILE = os.path.join(PathManager.BASE, "best_config.json")

PHASE_META = {
    1: dict(subdir="s1_search", csv="s1_results.csv", label="Stage 1 Search"),
    2: dict(subdir="s2_search", csv="s2_results.csv", label="Stage 2 Search"),
}

BGE_ICL_QUERY_PREFIX = (
    "Given a humor retrieval query, find relevant jokes, puns, or wordplay. "
    "Query: "
)

EMBEDDER_SEARCH_SUBDIR = "embedder_search"
EMBEDDER_SEARCH_CSV = "embedder_results.csv"


@dataclass
class RunConfig:
    """Unified config for all phases."""
    # Stage 1
    stage1_topk: int = 2000
    bm25_k1: float = 1.2
    bm25_b: float = 0.5
    rrf_k: int = 60
    rrf_bm25_weight: float = 1.0
    query_expansion_variants: int = 3
    # Stage 2
    stage2_topk: int = 500
    stage2_ce_blend: float = 0.70
    stage2_dedup: float = 0.97
    # Meta
    ce_name: str = "none"
    group: str = ""

    def _s1_id(self) -> str:
        return (f"tk{self.stage1_topk}"
                f"_k{self.bm25_k1:.1f}_b{self.bm25_b:.2f}"
                f"_rk{self.rrf_k}_bw{self.rrf_bm25_weight:.1f}"
                f"_qe{self.query_expansion_variants}")

    def _s2_id(self) -> str:
        return (f"s2tk{self.stage2_topk}"
                f"_blend{self.stage2_ce_blend:.2f}"
                f"_ded{self.stage2_dedup:.2f}")

    def run_name(self, phase: int) -> str:
        prefix = {1: "S1", 2: "S2"}[phase]
        suffix = {1: self._s1_id(), 2: self._s2_id()}[phase]
        return f"{prefix}_{self.ce_name}__{suffix}"


def _load_best_config(required_phase: int) -> dict:
    if not os.path.exists(BEST_CONFIG_FILE):
        raise FileNotFoundError(
            f"{BEST_CONFIG_FILE} not found.\n"
            f"Run Phase {required_phase - 1} first:  python run_search.py --phase {required_phase - 1}"
        )
    with open(BEST_CONFIG_FILE) as f:
        cfg = json.load(f)
    recorded_phase = cfg.get("phase", 0)
    if recorded_phase < required_phase - 1:
        print(
            f"WARNING: best_config.json is from Phase {recorded_phase}. "
            f"Results may be suboptimal - run Phase {required_phase - 1} first."
        )
    return cfg


def _save_best_config(cfg: dict) -> None:
    with open(BEST_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\t[OK] best_config.json -> {BEST_CONFIG_FILE}")


def _build_s1_grid(ce_name: str) -> list[RunConfig]:
    keys = list(S1_GRID.keys())
    combos = itertools.product(*[S1_GRID[k] for k in keys])
    return [
        RunConfig(**dict(zip(keys, c)), **S2_DEFAULTS, ce_name=ce_name)
        for c in combos
    ]


def _build_s2_grid(s1: dict, ce_name: str) -> list[RunConfig]:
    keys = list(S2_GRID.keys())
    combos = itertools.product(*[S2_GRID[k] for k in keys])
    s1_fixed = dict(
        stage1_topk=s1["stage1_topk"],
        bm25_k1=s1["bm25_k1"], bm25_b=s1["bm25_b"],
        rrf_k=s1["rrf_k"], rrf_bm25_weight=s1["rrf_bm25_weight"],
        query_expansion_variants=s1["query_expansion_variants"],
    )
    return [
        RunConfig(**s1_fixed, **dict(zip(keys, c)), ce_name=ce_name)
        for c in combos
    ]


def _stage1(query_text: str, q_emb: torch.Tensor,
            corpus_data: CorpusData, cfg: RunConfig) -> np.ndarray:
    n_docs = len(corpus_data.corpus_texts)
    per_sys = max(cfg.stage1_topk * 2, 2000)

    expansions = expand_query(query_text)[:cfg.query_expansion_variants]
    bm25_rankings = []
    for eq in expansions:
        scores = corpus_data.bm25.get_scores(tokenize_stemmed(eq))
        idx = np.argpartition(scores, -per_sys)[-per_sys:]
        bm25_rankings.append(idx[np.argsort(scores[idx])[::-1]])

    bm25_fused = _rrf_fuse([(r, 1.0) for r in bm25_rankings], n_docs, k=cfg.rrf_k)
    bm25_top = np.argpartition(bm25_fused, -per_sys)[-per_sys:]
    bm25_top = bm25_top[np.argsort(bm25_fused[bm25_top])[::-1]]

    dense_scores = util.cos_sim(q_emb, corpus_data.corpus_embeddings)[0].cpu().numpy()
    dense_top = np.argpartition(dense_scores, -per_sys)[-per_sys:]
    dense_top = dense_top[np.argsort(dense_scores[dense_top])[::-1]]

    rankings = [
        (bm25_top, cfg.rrf_bm25_weight),
        (dense_top, 1.0),
    ]

    fused = _rrf_fuse(rankings, n_docs, k=cfg.rrf_k)
    top = np.argpartition(fused, -cfg.stage1_topk)[-cfg.stage1_topk:]
    return top[np.argsort(fused[top])[::-1]]


def _stage2(query_text: str, top1: np.ndarray,
            corpus_data: CorpusData, ce: CrossEncoder | None,
            cfg: RunConfig) -> list[tuple[int, float]]:
    cands = _dedup_candidates(top1, corpus_data.corpus_embeddings, cfg.stage2_dedup)

    if ce is None:
        keep = cands[:cfg.stage2_topk]
        return [(int(i), float(len(keep) - r)) for r, i in enumerate(keep)]

    pairs = [[query_text, corpus_data.corpus_texts[i]] for i in cands]
    try:
        n_params = sum(p.numel() for p in ce.model.parameters())
    except Exception:
        n_params = 0
    bs = 16 if n_params > 200_000_000 else (64 if n_params > 50_000_000 else 256)

    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            ce_raw = ce.predict(pairs, batch_size=bs)
    else:
        ce_raw = ce.predict(pairs, batch_size=bs)
    ce_raw = np.asarray(ce_raw, dtype=np.float64)

    n = len(cands)
    ce_order = np.argsort(ce_raw)[::-1]
    s1_order = np.arange(n)
    fused = np.zeros(n, dtype=np.float64)
    rk = cfg.rrf_k + np.arange(1, n + 1)
    fused[ce_order] += cfg.stage2_ce_blend / rk
    fused[s1_order] += (1 - cfg.stage2_ce_blend) / rk

    final = np.argsort(fused)[::-1][:cfg.stage2_topk]
    return [(int(cands[i]), float(fused[i])) for i in final]


def _stage3_no_judge(ranked: list[tuple[int, float]],
                     corpus_data: CorpusData, cfg: RunConfig) -> list[dict]:
    """Pure pass-through stage 3 (no judge, no prior blend)."""
    scored = [{"docid": corpus_data.corpus_docids[doc_idx], "score": s2_score} for doc_idx, s2_score in ranked]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _precompute_s1(
        queries: list[dict],
        q_embs_cpu: torch.Tensor,
        corpus_data: CorpusData,
        cfg: RunConfig,
) -> dict[str, np.ndarray]:
    """
    Compute Stage 1 for every query, return {qid -> top1_indices}.
    Embeddings are moved from CPU to GPU one at a time to stay within 6GB VRAM while the CE model is also loaded.
    """
    device = corpus_data.corpus_embeddings.device
    cache: dict[str, np.ndarray] = {}
    for qi, qobj in enumerate(queries):
        cache[str(qobj["qid"])] = _stage1(
            qobj["query"], q_embs_cpu[qi].to(device), corpus_data, cfg
        )
    return cache


def _precompute_s2(
        queries: list[dict],
        s1_cache: dict[str, np.ndarray],
        corpus_data: CorpusData,
        ce: CrossEncoder | None,
        cfg: RunConfig,
) -> dict[str, list[tuple[int, float]]]:
    """
    Compute Stage 2 (CE reranking) for every query, return {qid -> ranked list}.
    With a single CE across all runs, Stage 2 only needs recomputing when Stage 1 changes (i.e. the S1 key changes).
    """
    cache: dict[str, list[tuple[int, float]]] = {}
    for qi, qobj in enumerate(queries):
        qid = str(qobj["qid"])
        cache[qid] = _stage2(qobj["query"], s1_cache[qid], corpus_data, ce, cfg)
        if (qi + 1) % 10 == 0:
            print(f"\t\tS2 {qi + 1}/{len(queries)} queries...")
    return cache


def _run_pipeline(run_name: str, queries: list[dict], qrel_dict: dict,
                  corpus_data: CorpusData, ce: CrossEncoder | None,
                  cfg: RunConfig, phase: int,
                  s1_cache: dict | None = None,
                  s2_cache: dict | None = None,
                  q_embs_cpu: torch.Tensor | None = None,
                  ) -> tuple[list[dict], dict]:
    """
    Run the full pipeline for a set of queries.

    When s1_cache / s2_cache are provided the corresponding stages are skipped entirely.
    Only Stage 3 (no-judge score blend) runs per combo.

    Phase 1 - S1 key changes per combo. S2 also re-runs (CE output depends on S1).
                Query embeddings encoded once, passed in via q_embs_cpu.
    Phase 2 - S1 fixed. S2 changes (blend/dedup/topk vary). S1 computed once.
    Phase 3 - S1+S2 fixed for s3_only/baseline groups with both caches reused.
                rrf_only/combined change S1, so they get a fresh S1+S2 compute.

    q_embs_cpu stays on CPU. Moved per-query to avoid OOM on 6GB VRAM.
    """
    submission, recalls = [], {1: [], 2: [], 3: []}
    device = corpus_data.corpus_embeddings.device

    if q_embs_cpu is None and s1_cache is None:
        q_texts = [q["query"] for q in queries]
        q_embs_cpu = corpus_data.embedder.encode(
            q_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=False,
        ).cpu()

    for qi, qobj in enumerate(queries):
        qid = str(qobj["qid"])
        qtxt = qobj["query"]
        relevant = qrel_dict.get(qid, set())

        if s1_cache is not None:
            top1 = s1_cache[qid]
        else:
            top1 = _stage1(qtxt, q_embs_cpu[qi].to(device), corpus_data, cfg)
        if relevant:
            recalls[1].append(len(relevant & {corpus_data.corpus_docids[i] for i in top1}) / len(relevant))

        if s2_cache is not None:
            ranked = s2_cache[qid]
        else:
            ranked = _stage2(qtxt, top1, corpus_data, ce, cfg)
        if relevant:
            s2ids = {corpus_data.corpus_docids[i] for i, _ in ranked}
            recalls[2].append(len(relevant & s2ids) / len(relevant))

        scored = _stage3_no_judge(ranked, corpus_data, cfg)
        if relevant:
            s3ids = {d["docid"] for d in scored}
            recalls[3].append(len(relevant & s3ids) / len(relevant))

        for rank, item in enumerate(scored):
            submission.append({
                "run_id": run_name, "manual": 0,
                "qid": int(qid), "docid": str(item["docid"]),
                "rank": rank, "score": round(float(item["score"]), 6),
            })

        if (qi + 1) % 10 == 0:
            print(f"\t\t{qi + 1}/{len(queries)} queries...")

    return submission, recalls


def _build_bm25(corpus, corpus_texts, corpus_docids, k1: float, b: float, use_expanded: bool) -> BM25Okapi:
    if use_expanded and os.path.exists(PathManager.RATIONALES_FILE):
        rat = {
            str(x.get("docid", "")): x.get("rationale", "") for x in load_json(PathManager.RATIONALES_FILE)
            if x.get("docid") and x.get("rationale")
        }
        docs = [
            f"{corpus_texts[i]} {rat[corpus_docids[i]]}" if corpus_docids[i] in rat else corpus_texts[i]
            for i in range(len(corpus))
        ]
    else:
        docs = corpus_texts
    return BM25Okapi([tokenize_stemmed(t) for t in docs], k1=k1, b=b)


def _make_corpus_data(
        corpus_texts, corpus_docids, embedder, corpus_embeddings, bm25, humor_prior=None, humor_prior_array=None
) -> CorpusData:
    n = len(corpus_docids)
    return CorpusData(
        corpus_texts=corpus_texts,
        corpus_docids=corpus_docids,
        bm25=bm25,
        embedder=embedder,
        corpus_embeddings=corpus_embeddings,
        humor_prior=humor_prior or {},
        humor_prior_array=humor_prior_array if humor_prior_array is not None else np.full(n, 0.5),
    )


def _resolve_ce(args_ce: str | None, fallback_cfg: dict | None = None) -> str | None:
    if args_ce:
        return args_ce
    if fallback_cfg and fallback_cfg.get("ce_name"):
        return fallback_cfg["ce_name"]
    found = next((n for n in CE_PRIORITY if os.path.exists(os.path.join(PathManager.MODELS_DIR, n))), None)
    if not found:
        raise RuntimeError("No CE model found on disk. Train one first.")
    return found


def _record_from_metrics(
        run_name: str, cfg: RunConfig, phase: int, metrics: dict, recalls: dict, elapsed: float
) -> dict:
    return {
        "run_name": run_name,
        "ce": cfg.ce_name,
        "stage1_topk": cfg.stage1_topk,
        "bm25_k1": cfg.bm25_k1,
        "bm25_b": cfg.bm25_b,
        "rrf_k": cfg.rrf_k,
        "rrf_bm25_weight": cfg.rrf_bm25_weight,
        "query_expansion_variants": cfg.query_expansion_variants,
        "stage2_topk": cfg.stage2_topk,
        "stage2_ce_blend": cfg.stage2_ce_blend,
        "stage2_dedup": cfg.stage2_dedup,
        "MAP": round(metrics.get("map", 0), 4),
        "R@30": round(metrics.get("recall_30", 0), 4),
        "R@1000": round(metrics.get("recall_1000", 0), 4),
        "NDCG@10": round(metrics.get("ndcg_cut_10", 0), 4),
        "P@5": round(metrics.get("P_5", 0), 4),
        "P@10": round(metrics.get("P_10", 0), 4),
        **stage_recall_record(recalls),
        "time_sec": round(elapsed, 1),
    }


def _generate_submissions(
        phase: int, top_n: int, corpus, corpus_texts, corpus_docids,
        embedder, corpus_embeddings, humor_prior, humor_prior_array
) -> None:
    meta = PHASE_META[phase]
    out_dir = os.path.join(PathManager.RESULTS_DIR, meta["subdir"])
    done_csv = os.path.join(out_dir, meta["csv"])

    if not os.path.exists(done_csv):
        print(f"ERROR: {done_csv} not found - run the search first.")
        return

    df = pd.read_csv(done_csv).sort_values("MAP", ascending=False).reset_index(drop=True)

    sub_dir = os.path.join(out_dir, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    tracker_csv = os.path.join(sub_dir, "submission_tracker.csv")

    done_subs: set[str] = set()
    tracker_rows: list[dict] = []
    if os.path.exists(tracker_csv):
        tdf = pd.read_csv(tracker_csv)
        done_subs = set(tdf["run_name"].tolist())
        tracker_rows = tdf.to_dict("records")
        print(f"\t{len(done_subs)} already done, skipping")

    pending_rows = []
    for _, row in df.iterrows():
        sub_rname = f"VANGUARD_Task1_{row['run_name']}"
        if sub_rname not in done_subs:
            pending_rows.append(row)
        if len(pending_rows) == top_n:
            break

    if not pending_rows:
        print(f"\tAll top-{top_n} configs already submitted... nothing to do.")
        return

    print(f"\nGenerating {len(pending_rows)} CodaBench submissions (Phase {phase}):")
    print("-" * 70)
    for i, row in enumerate(pending_rows):
        extra = f"\t[{row['group']}]" if "group" in row else ""
        print(f"\t{i + 1:>3}. MAP={row['MAP']:.4f}{extra}  {row['run_name']}")
    print("-" * 70)

    test_queries = load_json(PathManager.QUERIES_TEST_FILE)
    print(f"\tOfficial test queries: {len(test_queries)}")

    bm25_cache: dict[tuple, BM25Okapi] = {}
    ce_cache: dict[str, CrossEncoder] = {}

    for rank_i, row in enumerate(pending_rows, 1):
        sub_rname = f"VANGUARD_Task1_{row['run_name']}"

        ce_name = str(row["ce"])
        if ce_name not in ce_cache:
            print(f"\n\tLoading CE: {ce_name}...")
            ce_cache[ce_name] = CrossEncoder(os.path.join(PathManager.MODELS_DIR, ce_name))

        cfg = RunConfig(
            stage1_topk=int(row["stage1_topk"]),
            bm25_k1=float(row["bm25_k1"]),
            bm25_b=float(row["bm25_b"]),
            rrf_k=int(row["rrf_k"]),
            rrf_bm25_weight=float(row["rrf_bm25_weight"]),
            query_expansion_variants=int(row["query_expansion_variants"]),
            stage2_topk=int(row["stage2_topk"]),
            stage2_ce_blend=float(row["stage2_ce_blend"]),
            stage2_dedup=float(row["stage2_dedup"]),
            ce_name=ce_name,
            group=str(row.get("group", "")),
        )

        bm25_key = (cfg.bm25_k1, cfg.bm25_b)
        if bm25_key not in bm25_cache:
            print(f"\tBuilding BM25 (k1={cfg.bm25_k1}, b={cfg.bm25_b})...")
            bm25_cache[bm25_key] = _build_bm25(
                corpus, corpus_texts, corpus_docids, cfg.bm25_k1, cfg.bm25_b, USE_EXPANDED_BM25,
            )

        corpus_data = _make_corpus_data(
            corpus_texts, corpus_docids, embedder, corpus_embeddings,
            bm25_cache[bm25_key], humor_prior, humor_prior_array,
        )

        print(f"\n\t{'-' * 58}")
        print(f"\t[{rank_i}/{len(pending_rows)}] {sub_rname}")
        t0 = time.time()
        sub_data, _ = _run_pipeline(sub_rname, test_queries, {}, corpus_data, ce_cache[ce_name], cfg, phase)
        elapsed = time.time() - t0

        sub_path = os.path.join(sub_dir, f"prediction_{sub_rname}.json")
        save_json(sub_data, sub_path)
        n_q = len({x["qid"] for x in sub_data})
        print(f"\tSaved: {n_q} queries, {len(sub_data)} records  ({elapsed:.0f}s)")

        tracker_rows.append({
            "run_name": sub_rname, "ce": ce_name, "phase": phase,
            f"phase{phase}_MAP": round(float(row["MAP"]), 4),
            "group": row.get("group", ""), "num_queries": n_q,
            "num_records": len(sub_data), "time_sec": round(elapsed, 1),
        })
        done_subs.add(sub_rname)
        pd.DataFrame(tracker_rows).to_csv(tracker_csv, index=False)

    best_rname = f"VANGUARD_Task1_{df.iloc[0]['run_name']}"
    best_path = os.path.join(sub_dir, f"prediction_{best_rname}.json")
    if os.path.exists(best_path):
        dst_json = os.path.join(PathManager.BASE, f"{best_rname}.json")
        dst_zip = os.path.join(PathManager.BASE, "prediction.zip")
        shutil.copy(best_path, dst_json)
        with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(dst_json, f"{best_rname}.json")
        next_hint = (
            f"\tNext: python run_search.py --phase {phase + 1}" if phase < 3 else "  Done. Add judge on top of config."
        )
        print(f"\n{'=' * 70}")
        print(f"\tTOP-{top_n} SUBMISSIONS READY  (Phase {phase})")
        print(f"\tBest MAP={df.iloc[0]['MAP']:.4f}  ->  {best_rname}")
        print(f"\tprediction.zip  -> {dst_zip}")
        print(f"\tUpload to CodaBench, check results.")
        print(next_hint)
        print(f"{'=' * 70}")


def _run_phase(phase: int, args) -> None:
    """Core search loop shared by all three phases."""
    seed_everything()
    ensure_dirs()

    meta = PHASE_META[phase]
    out_dir = os.path.join(PathManager.RESULTS_DIR, meta["subdir"])
    os.makedirs(out_dir, exist_ok=True)
    done_csv = os.path.join(out_dir, meta["csv"])

    if phase == 1:
        prev_cfg = {}
        ce_name = _resolve_ce(args.ce)
        grid = _build_s1_grid(ce_name)
    elif phase == 2:
        prev_cfg = _load_best_config(required_phase=2)
        ce_name = _resolve_ce(args.ce, prev_cfg)
        grid = _build_s2_grid(prev_cfg, ce_name)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    done_runs: set[str] = set()
    all_results: list[dict] = []
    if os.path.exists(done_csv):
        df_done = pd.read_csv(done_csv)
        done_runs = set(df_done["run_name"].tolist())
        all_results = df_done.to_dict("records")
        print(f"\tResuming - {len(done_runs)} runs already done.")

    remaining = [cfg for cfg in grid if cfg.run_name(phase) not in done_runs]

    print(f"Grid: {len(grid)} combos  |  Remaining: {len(remaining)}  |  CE: {ce_name}")

    if args.dry_run:
        print("\n[DRY RUN]")
        for cfg in remaining[:6]:
            extra = f"\t[{cfg.group}]" if phase == 3 else ""
            print(f"\tWould run: {cfg.run_name(phase)}{extra}")
        if len(remaining) > 6:
            print(f"\t... and {len(remaining) - 6} more")
        return

    print("\nLoading corpus...")
    corpus, corpus_texts, corpus_docids = load_corpus(PathManager.CORPUS_FILE)
    print("Loading dense embeddings...")
    embedder, corpus_embeddings = load_dense_embeddings(corpus_texts)

    print("Loading humor prior...")
    humor_prior, humor_prior_array = load_humor_prior(corpus_docids)
    if not humor_prior:
        print("  WARNING: No humor scores - run score_corpus_humor.py first.")

    test_queries = load_json(PathManager.LOCAL_TEST_QUERIES)
    qrels_data = load_json(PathManager.LOCAL_TEST_QRELS)
    qrel_dict = build_qrel_dict(qrels_data)
    print(f"Queries: {len(test_queries)}  |  Qrels: {len(qrels_data)}")

    print(f"Loading CE: {ce_name}...")
    ce = CrossEncoder(os.path.join(PathManager.MODELS_DIR, ce_name))

    print("Pre-encoding queries (once for all combos)...")
    q_texts = [q["query"] for q in test_queries]
    q_embs_cpu = embedder.encode(q_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=False).cpu()

    s1_result_cache: dict[tuple, dict[str, np.ndarray]] = {}
    s2_result_cache: dict[tuple, dict[str, list]] = {}
    bm25_cache: dict[tuple, BM25Okapi] = {}

    for i, cfg in enumerate(remaining, 1):
        rname = cfg.run_name(phase)
        print(f"\n[{i}/{len(remaining)}]  {rname}")

        bm25_key = (cfg.bm25_k1, cfg.bm25_b)
        if bm25_key not in bm25_cache:
            print(f"\tBuilding BM25 (k1={cfg.bm25_k1}, b={cfg.bm25_b})...")
            bm25_cache[bm25_key] = _build_bm25(
                corpus, corpus_texts, corpus_docids, cfg.bm25_k1, cfg.bm25_b, USE_EXPANDED_BM25,
            )

        corpus_data = _make_corpus_data(
            corpus_texts, corpus_docids, embedder, corpus_embeddings,
            bm25_cache[bm25_key], humor_prior, humor_prior_array,
        )

        s1_key = (
            cfg.stage1_topk, cfg.bm25_k1, cfg.bm25_b, cfg.rrf_k, cfg.rrf_bm25_weight, cfg.query_expansion_variants
        )
        if s1_key not in s1_result_cache:
            print(f"\tComputing Stage 1 (new key)...")
            s1_result_cache[s1_key] = _precompute_s1(test_queries, q_embs_cpu, corpus_data, cfg)
        else:
            print(f"\tStage 1: cache hit")
        s1_cache = s1_result_cache[s1_key]

        s2_key = s1_key + (cfg.stage2_topk, cfg.stage2_ce_blend, cfg.stage2_dedup)
        if s2_key not in s2_result_cache:
            print(f"\tComputing Stage 2 (new key)...")
            s2_result_cache[s2_key] = _precompute_s2(test_queries, s1_cache, corpus_data, ce, cfg)
        else:
            print(f"\tStage 2: cache hit")
        s2_cache = s2_result_cache[s2_key]

        t0 = time.time()
        sub_data, recalls = _run_pipeline(
            rname, test_queries, qrel_dict, corpus_data, ce, cfg, phase, s1_cache=s1_cache, s2_cache=s2_cache,
        )
        elapsed = time.time() - t0
        metrics = evaluate_trec(sub_data, qrels_data)

        record = _record_from_metrics(rname, cfg, phase, metrics, recalls, elapsed)
        all_results.append(record)
        done_runs.add(rname)

        print(
            f"\tMAP={record['MAP']:.4f}  NDCG@10={record['NDCG@10']:.4f}  R@30={record['R@30']:.4f}  ({elapsed:.0f}s)"
        )

        pd.DataFrame(all_results).sort_values("MAP", ascending=False).to_csv(done_csv, index=False)

    del ce
    torch.cuda.empty_cache()

    _print_phase_summary(phase, all_results, prev_cfg)
    _update_best_config(phase, all_results, ce_name, prev_cfg)

    print(f"\nGenerating top-{args.top_n} CodaBench submissions...")
    _generate_submissions(
        phase=phase, top_n=args.top_n,
        corpus=corpus, corpus_texts=corpus_texts, corpus_docids=corpus_docids,
        embedder=embedder, corpus_embeddings=corpus_embeddings,
        humor_prior=humor_prior, humor_prior_array=humor_prior_array,
    )


def _submit_only(phase: int, args) -> None:
    """Skip the search. Just regenerate submissions from an existing CSV."""
    seed_everything()
    ensure_dirs()

    print("\nLoading corpus...")
    corpus, corpus_texts, corpus_docids = load_corpus(PathManager.CORPUS_FILE)
    print("Loading dense embeddings...")
    embedder, corpus_embeddings = load_dense_embeddings(corpus_texts)

    humor_prior, humor_prior_array = load_humor_prior(corpus_docids)

    _generate_submissions(
        phase=phase, top_n=args.top_n,
        corpus=corpus, corpus_texts=corpus_texts, corpus_docids=corpus_docids,
        embedder=embedder, corpus_embeddings=corpus_embeddings,
        humor_prior=humor_prior, humor_prior_array=humor_prior_array,
    )


def _print_phase_summary(phase: int, results: list[dict], prev_cfg: dict) -> None:
    df = pd.DataFrame(results).sort_values("MAP", ascending=False).reset_index(drop=True)
    print(f"\n{'=' * 70}")
    print(f"PHASE {phase} COMPLETE - {len(df)} configurations  ({PHASE_META[phase]['label']})")
    print(f"{'=' * 70}")

    print(f"\n{'Rank':<5} {'MAP':>6} {'NDCG@10':>8} {'R@30':>6}  Config")
    print("-" * 70)
    for i, row in df.head(15).iterrows():
        print(f"\t{i + 1:<3} {row['MAP']:>6.4f} {row['NDCG@10']:>8.4f} "
              f"{row['R@30']:>6.4f}  {row['run_name']}")

    param_key = {
        1: ["stage1_topk", "bm25_k1", "bm25_b", "rrf_k", "rrf_bm25_weight", "query_expansion_variants"],
        2: ["stage2_topk", "stage2_ce_blend", "stage2_dedup"],
    }[phase]
    print("\n-- Parameter sensitivity (best MAP per value) --")
    for p in param_key:
        best = df.groupby(p)["MAP"].max()
        parts = "  |  ".join(f"{v}->{m:.4f}" for v, m in best.items())
        print(f"\t{p:<30}: {parts}")


def _update_best_config(phase: int, results: list[dict], ce_name: str, prev_cfg: dict) -> None:
    best = max(results, key=lambda x: x["MAP"])

    cfg = dict(prev_cfg)
    cfg["phase"] = phase
    cfg["ce_name"] = ce_name

    if phase == 1:
        cfg.update({
            "stage1_topk": best["stage1_topk"],
            "bm25_k1": best["bm25_k1"],
            "bm25_b": best["bm25_b"],
            "rrf_k": best["rrf_k"],
            "rrf_bm25_weight": best["rrf_bm25_weight"],
            "query_expansion_variants": best["query_expansion_variants"],
            "stage2_topk": S2_DEFAULTS["stage2_topk"],
            "stage2_ce_blend": S2_DEFAULTS["stage2_ce_blend"],
            "stage2_dedup": S2_DEFAULTS["stage2_dedup"],
            "phase1_MAP": best["MAP"],
            "phase1_NDCG10": best["NDCG@10"],
        })
    elif phase == 2:
        cfg.update({
            "stage2_topk": best["stage2_topk"],
            "stage2_ce_blend": best["stage2_ce_blend"],
            "stage2_dedup": best["stage2_dedup"],
            "phase2_MAP": best["MAP"],
            "phase2_NDCG10": best["NDCG@10"],
        })

    _save_best_config(cfg)
    prev_map = prev_cfg.get(f"phase{phase - 1}_MAP", None)
    delta_str = f"\tΔ={best['MAP'] - prev_map:+.4f}" if prev_map else ""
    print(f"\tPhase {phase} MAP={best['MAP']:.4f}{delta_str}")
    print(
        f"\n  Next step: python run_search.py --phase {phase + 1}"
        if phase < 2 else f"\n  Best S1+S2 config locked in best_config.json."
    )


def _load_embedder_and_cache(emb_cfg: dict, corpus_texts: list[str]) -> tuple[SentenceTransformer, torch.Tensor]:
    """
    Load (or recompute and cache) corpus embeddings for one embedder candidate.
    cache_file in emb_cfg is an absolute path (defined in config.py).
    The bge-base cache is shared with the main pipeline and never recomputed if it already exists.
    """
    cache_path = emb_cfg["cache_file"]  # already absolute (from config.py)
    embedder = SentenceTransformer(emb_cfg["model_id"], device="cpu")

    if os.path.exists(cache_path):
        corpus_embeddings = torch.load(cache_path, weights_only=True, map_location="cpu").float()
        print(f"\t\tLoaded cache: {cache_path}  {corpus_embeddings.shape}")
    else:
        print(f"\t\tComputing corpus embeddings for {emb_cfg['name']} "
              f"({len(corpus_texts)} docs) - this may take a few minutes...")
        corpus_embeddings = embedder.encode(
            corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=64
        ).cpu()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(corpus_embeddings, cache_path)
        print(f"\t\tSaved -> {cache_path}")

    return embedder, corpus_embeddings


def _encode_queries_for_embedder(
        embedder: SentenceTransformer,
        query_texts: list[str],
        icl: bool,
        corpus_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Encode queries, optionally prepending the ICL task prefix for bge-en-icl.
    Moves the result to the same device as corpus_embeddings.
    """
    if icl:
        prefixed = [BGE_ICL_QUERY_PREFIX + q for q in query_texts]
    else:
        prefixed = query_texts

    q_embs = embedder.encode(prefixed, convert_to_tensor=True, batch_size=16, show_progress_bar=False)
    q_embs = q_embs.float()
    return q_embs.to(corpus_embeddings.device)


def run_embedder_search(args) -> None:
    """
    Fix all pipeline params at the best known config (from best_config.json or the hardcoded known-good values if
    the file does not exist yet), vary only the dense embedder, and compare Stage 1 recall + final MAP for each.

    CE is fixed to the best model (GTE-Reranker-ModernBERT-Base without augmentation, or whatever --ce specifies).

    Results are written to results/embedder_search/embedder_results.csv.
    No CodaBench submissions are generated. This is a diagnostic comparison and not a submission run.
    Take the winning embedder name and plug it into DENSE_MODEL in config.py for subsequent phases.
    """
    seed_everything()
    ensure_dirs()

    out_dir = os.path.join(PathManager.RESULTS_DIR, EMBEDDER_SEARCH_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    done_csv = os.path.join(out_dir, EMBEDDER_SEARCH_CSV)

    done_names: set[str] = set()
    all_results: list[dict] = []
    if os.path.exists(done_csv):
        df_done = pd.read_csv(done_csv)
        done_names = set(df_done["embedder"].tolist())
        all_results = df_done.to_dict("records")
        print(f"\tResuming - {len(done_names)} embedders already done.")

    fixed = dict(KNOWN_GOOD_CONFIG)
    print(f"\tPipeline params: KNOWN_GOOD_CONFIG "
          f"(k1={fixed['bm25_k1']}, b={fixed['bm25_b']}, "
          f"blend={fixed['stage2_ce_blend']}, topk={fixed['stage1_topk']})")

    cfg = RunConfig(
        stage1_topk=fixed["stage1_topk"],
        bm25_k1=fixed["bm25_k1"],
        bm25_b=fixed["bm25_b"],
        rrf_k=fixed["rrf_k"],
        rrf_bm25_weight=fixed["rrf_bm25_weight"],
        query_expansion_variants=fixed["query_expansion_variants"],
        stage2_topk=fixed["stage2_topk"],
        stage2_ce_blend=fixed["stage2_ce_blend"],
        stage2_dedup=fixed["stage2_dedup"],
    )

    ce_name = _resolve_ce(args.ce, fixed)
    cfg.ce_name = ce_name

    remaining = [e for e in EMBEDDER_CANDIDATES if e["name"] not in done_names]

    print(f"\nEmbedder comparison - {len(EMBEDDER_CANDIDATES)} candidates | {len(remaining)} remaining | CE: {ce_name}")
    print("-" * 70)
    for e in EMBEDDER_CANDIDATES:
        marker = "[OK] done" if e["name"] in done_names else "[...] pending"
        print(f"\t{marker}  {e['name']:<14}  {e['description']}")
    print("-" * 70)

    if args.dry_run:
        print("\n[DRY RUN] - nothing will execute.")
        return

    print("\nLoading corpus...")
    corpus, corpus_texts, corpus_docids = load_corpus(PathManager.CORPUS_FILE)

    print("Building BM25 index...")
    bm25 = _build_bm25(corpus, corpus_texts, corpus_docids, cfg.bm25_k1, cfg.bm25_b, USE_EXPANDED_BM25)

    test_queries = load_json(PathManager.LOCAL_TEST_QUERIES)
    qrels_data = load_json(PathManager.LOCAL_TEST_QRELS)
    qrel_dict = build_qrel_dict(qrels_data)
    print(f"Queries: {len(test_queries)}  |  Qrels: {len(qrels_data)}")

    print(f"Loading CE: {ce_name}...")
    ce = CrossEncoder(os.path.join(PathManager.MODELS_DIR, ce_name))

    for emb_cfg in remaining:
        print(f"\n{'=' * 70}")
        print(f"\tEmbedder: {emb_cfg['name']}  ({emb_cfg['description']})")
        print(f"\tModel:    {emb_cfg['model_id']}")
        print(f"{'=' * 70}")

        try:
            embedder, corpus_embeddings = _load_embedder_and_cache(emb_cfg, corpus_texts)
        except Exception as e:
            print(f"\n\t[SKIP] Failed to load {emb_cfg['name']}: {e}")
            print(f"\tTip: pre-download with:")
            print(f"\t\thuggingface-cli download {emb_cfg['model_id']} --local-dir <cache_dir>")
            print(f"\tOr set HF_HUB_DISABLE_XET=1 to avoid the xet backend.")
            print(f"\tRe-run after the model is available - completed embedders are saved.\n")
            continue

        corpus_data = _make_corpus_data(corpus_texts, corpus_docids, embedder, corpus_embeddings, bm25)

        query_texts = [q["query"] for q in test_queries]
        print(f"\tEncoding {len(query_texts)} queries" + (" (with ICL prefix)" if emb_cfg["icl"] else "") + "...")
        try:
            q_embs = _encode_queries_for_embedder(embedder, query_texts, emb_cfg["icl"], corpus_embeddings)
        except Exception as e:
            print(f"\n\t[SKIP] Query encoding failed for {emb_cfg['name']}: {e}")
            del embedder, corpus_embeddings
            torch.cuda.empty_cache()
            continue

        submission: list[dict] = []
        recalls: dict[int, list[float]] = {1: [], 2: [], 3: []}

        t0 = time.time()
        for qi, qobj in enumerate(test_queries):
            qid = str(qobj["qid"])
            qtxt = qobj["query"]
            relevant = qrel_dict.get(qid, set())

            top1 = _stage1(qtxt, q_embs[qi], corpus_data, cfg)
            if relevant:
                recalls[1].append(len(relevant & {corpus_data.corpus_docids[i] for i in top1}) / len(relevant))

            ranked = _stage2(qtxt, top1, corpus_data, ce, cfg)
            if relevant:
                s2ids = {corpus_data.corpus_docids[i] for i, _ in ranked}
                r2 = len(relevant & s2ids) / len(relevant)
                recalls[2].append(r2)
                recalls[3].append(r2)  # no judge

            scored = _stage3_no_judge(ranked, corpus_data, cfg)
            for rank, item in enumerate(scored):
                submission.append({
                    "run_id": f"EMB_{emb_cfg['name']}", "manual": 0,
                    "qid": int(qid), "docid": str(item["docid"]),
                    "rank": rank, "score": round(float(item["score"]), 6),
                })

            if (qi + 1) % 10 == 0:
                print(f"\t\t{qi + 1}/{len(test_queries)} queries...")

        elapsed = time.time() - t0
        metrics = evaluate_trec(submission, qrels_data)

        from utils import safe_mean
        record = {
            "embedder": emb_cfg["name"],
            "model_id": emb_cfg["model_id"],
            "icl": emb_cfg["icl"],
            "ce": ce_name,
            "MAP": round(metrics.get("map", 0), 4),
            "R@30": round(metrics.get("recall_30", 0), 4),
            "R@1000": round(metrics.get("recall_1000", 0), 4),
            "NDCG@10": round(metrics.get("ndcg_cut_10", 0), 4),
            "P@5": round(metrics.get("P_5", 0), 4),
            "P@10": round(metrics.get("P_10", 0), 4),
            "s1_recall": round(safe_mean(recalls[1]), 4),
            "s2_recall": round(safe_mean(recalls[2]), 4),
            "time_sec": round(elapsed, 1),
            "description": emb_cfg["description"],
        }
        all_results.append(record)
        done_names.add(emb_cfg["name"])

        print(f"\tMAP={record['MAP']:.4f}  NDCG@10={record['NDCG@10']:.4f}  "
              f"R@30={record['R@30']:.4f}  R@1000={record['R@1000']:.4f}")
        print(f"\tS1 recall={record['s1_recall']:.4f}  S2 recall={record['s2_recall']:.4f}  ({elapsed:.0f}s)")

        pd.DataFrame(all_results).sort_values("MAP", ascending=False).to_csv(done_csv, index=False)

        del embedder, corpus_embeddings
        torch.cuda.empty_cache()

    del ce
    torch.cuda.empty_cache()

    if not all_results:
        return

    df = pd.DataFrame(all_results).sort_values("MAP", ascending=False).reset_index(drop=True)
    base = df[df["embedder"] == "bge-base"]
    base_map = float(base["MAP"].iloc[0]) if len(base) else 0.0

    print(f"\n{'=' * 70}")
    print(f"EMBEDDER COMPARISON COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n{'Rank':<5} {'Embedder':<14} {'MAP':>6} {'Δ MAP':>7} "
          f"{'NDCG@10':>8} {'R@30':>6} {'R@1K':>6} {'S1 rec':>7} {'S2 rec':>7}")
    print("-" * 75)
    for i, row in df.iterrows():
        delta = row["MAP"] - base_map
        marker = " <- baseline" if row["embedder"] == "bge-base" else ""
        print(f"\t{i + 1:<3} {row['embedder']:<14} {row['MAP']:>6.4f} "
              f"{delta:>+7.4f} {row['NDCG@10']:>8.4f} "
              f"{row['R@30']:>6.4f} {row['R@1000']:>6.4f} "
              f"{row['s1_recall']:>7.4f} {row['s2_recall']:>7.4f}{marker}")

    best = df.iloc[0]
    delta_best = best["MAP"] - base_map
    print(f"\n-- Verdict --")
    if delta_best > 0.002:
        print(f"\t[OK] {best['embedder']} improves MAP by {delta_best:+.4f} over bge-base.")
        print(f"\t\tSet DENSE_MODEL = \"{best['model_id']}\" in config.py")
        print(f"\t\tand delete the old corpus_emb_bge-base.pt cache.")
    elif delta_best > -0.002:
        print(f"\t[~] No embedder improves meaningfully over bge-base (Δ<0.002).")
        print(f"\t\tKeep DENSE_MODEL = \"BAAI/bge-base-en-v1.5\".")
    else:
        print(f"\t[x] All alternatives perform worse than bge-base on this task.")

    print(f"\n  Full results: {done_csv}")
    print(f"\tS1 recall column shows where the embedder difference originates -")
    print(f"\tif S1 recall is similar but MAP differs, the CE is compensating.")

    if not args.submit_only:
        _generate_embedder_submissions(df, ce_name, fixed, args.top_n)


def _generate_embedder_submissions(df: "pd.DataFrame", ce_name: str, fixed: dict, top_n: int) -> None:
    """
    Generate one CodaBench submission file per embedder (ranked by local MAP), package the best one as prediction.zip,
    and write a tracker CSV.

    Uses the official test queries (QUERIES_TEST_FILE) and not the local split.
    """
    seed_everything()
    out_dir = os.path.join(PathManager.RESULTS_DIR, EMBEDDER_SEARCH_SUBDIR)
    sub_dir = os.path.join(out_dir, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    tracker_csv = os.path.join(sub_dir, "submission_tracker.csv")

    done_subs: set[str] = set()
    tracker_rows: list[dict] = []
    if os.path.exists(tracker_csv):
        tdf = pd.read_csv(tracker_csv)
        done_subs = set(tdf["run_name"].tolist())
        tracker_rows = tdf.to_dict("records")
        print(f"\n\t{len(done_subs)} embedder submissions already done, skipping.")

    top_n = min(top_n, len(df))
    df_top = df.head(top_n)

    print(f"\nGenerating CodaBench submissions for top-{top_n} embedders:")
    print("-" * 70)
    for i, row in df_top.iterrows():
        print(f"\t{int(i) + 1:>3}. {row['embedder']:<14}  local MAP={row['MAP']:.4f}")
    print("-" * 70)

    print("\nLoading corpus & official test queries...")
    corpus, corpus_texts, corpus_docids = load_corpus(PathManager.CORPUS_FILE)
    test_queries = load_json(PathManager.QUERIES_TEST_FILE)
    print(f"\tOfficial test queries: {len(test_queries)}")

    print(f"Loading CE: {ce_name}...")
    ce = CrossEncoder(os.path.join(PathManager.MODELS_DIR, ce_name))

    bm25_key = (fixed["bm25_k1"], fixed["bm25_b"])
    print(f"Building BM25 (k1={bm25_key[0]}, b={bm25_key[1]})...")
    bm25 = _build_bm25(corpus, corpus_texts, corpus_docids, bm25_key[0], bm25_key[1], USE_EXPANDED_BM25)

    cfg = RunConfig(
        stage1_topk=fixed["stage1_topk"],
        bm25_k1=fixed["bm25_k1"],
        bm25_b=fixed["bm25_b"],
        rrf_k=fixed["rrf_k"],
        rrf_bm25_weight=fixed["rrf_bm25_weight"],
        query_expansion_variants=fixed["query_expansion_variants"],
        stage2_topk=fixed["stage2_topk"],
        stage2_ce_blend=fixed["stage2_ce_blend"],
        stage2_dedup=fixed["stage2_dedup"],
        ce_name=ce_name,
    )

    for rank_i, (_, row) in enumerate(df_top.iterrows(), 1):
        emb_name = str(row["embedder"])
        sub_rname = f"VANGUARD_Task1_EMB_{emb_name}"
        if sub_rname in done_subs:
            print(f"\t[{rank_i}/{top_n}] {sub_rname} - already done, skipping")
            continue

        emb_cfg = next((e for e in EMBEDDER_CANDIDATES if e["name"] == emb_name), None)
        if emb_cfg is None:
            print(f"\t[{rank_i}/{top_n}] {emb_name} - config not found, skipping")
            continue

        print(f"\n{'-' * 60}")
        print(f"\t[{rank_i}/{top_n}] {sub_rname}")
        try:
            embedder, corpus_embeddings = _load_embedder_and_cache(emb_cfg, corpus_texts)
        except Exception as e:
            print(f"\t[SKIP] Could not load {emb_name}: {e}")
            continue

        corpus_data = _make_corpus_data(corpus_texts, corpus_docids, embedder, corpus_embeddings, bm25)

        query_texts = [q["query"] for q in test_queries]
        print(f"\tEncoding {len(query_texts)} official queries" + (" (ICL prefix)" if emb_cfg["icl"] else "") + "...")
        try:
            q_embs = _encode_queries_for_embedder(embedder, query_texts, emb_cfg["icl"], corpus_embeddings)
        except Exception as e:
            print(f"\t[SKIP] Encoding failed for {emb_name}: {e}")
            del embedder, corpus_embeddings
            torch.cuda.empty_cache()
            continue

        submission: list[dict] = []
        t0 = time.time()
        for qi, qobj in enumerate(test_queries):
            qid = str(qobj["qid"])
            qtxt = qobj["query"]
            top1 = _stage1(qtxt, q_embs[qi], corpus_data, cfg)
            ranked = _stage2(qtxt, top1, corpus_data, ce, cfg)
            scored = _stage3_no_judge(ranked, corpus_data, cfg)
            for rank, item in enumerate(scored):
                submission.append({
                    "run_id": sub_rname, "manual": 0,
                    "qid": int(qid), "docid": str(item["docid"]),
                    "rank": rank, "score": round(float(item["score"]), 6),
                })
            if (qi + 1) % 50 == 0:
                print(f"\t\t{qi + 1}/{len(test_queries)} queries...")
        elapsed = time.time() - t0

        sub_path = os.path.join(sub_dir, f"prediction_{sub_rname}.json")
        save_json(submission, sub_path)
        n_q = len({x["qid"] for x in submission})
        print(f"\tSaved: {sub_path}  ({len(submission)} records, {n_q} queries, {elapsed:.0f}s)")

        tracker_rows.append({
            "run_name": sub_rname, "embedder": emb_name,
            "local_MAP": round(float(row["MAP"]), 4),
            "num_queries": n_q, "num_records": len(submission),
            "time_sec": round(elapsed, 1),
        })
        done_subs.add(sub_rname)
        pd.DataFrame(tracker_rows).to_csv(tracker_csv, index=False)

        del embedder, corpus_embeddings
        torch.cuda.empty_cache()

    del ce
    torch.cuda.empty_cache()

    if len(df_top):
        best_rname = f"VANGUARD_Task1_EMB_{df_top.iloc[0]['embedder']}"
        best_path = os.path.join(sub_dir, f"prediction_{best_rname}.json")
        if os.path.exists(best_path):
            dst_json = os.path.join(PathManager.BASE, f"{best_rname}.json")
            dst_zip = os.path.join(PathManager.BASE, "prediction.zip")
            shutil.copy(best_path, dst_json)
            with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(dst_json, f"{best_rname}.json")
            print(f"\n{'=' * 70}")
            print(f"\tEMBEDDER SUBMISSIONS READY")
            print(f"\tBest: {best_rname}  local MAP={df_top.iloc[0]['MAP']:.4f}")
            print(f"\tprediction.zip -> {dst_zip}")
            print(f"\tUpload to CodaBench!")
            print(f"\tAll {len(tracker_rows)} submissions in: {sub_dir}/")
            print(f"{'=' * 70}")


def main() -> None:
    """
    Two sequential phases in one script, each triggered by --phase N.
    Phases hand off via best_config.json in the project root.
        - Phase 1: Search Stage 1 params (BM25, RRF, topk, query expansion)
        - Phase 2: Fix best S1, search Stage 2 params (CE blend, dedup, topk)

    Each phase:
      - Runs local eval on the held-out split
      - Generates top-N CodaBench submission files + prediction.zip
      - Saves results to results/<phase_dir>/<phase>_results.csv
      - Updates best_config.json for the next phase

    Embedder comparison (--embedder-search)
    ----------------------------------------
    Separate mode that fixes pipeline params at the config from best_config.json and varies the dense embedder across:
      - bge-base-en-v1.5     (current baseline)
      - bge-large-en-v1.5    (larger)
      - bge-m3               (multi-granularity: dense + sparse + colbert)
      - bge-en-icl           (in-context learning, query-enriched encoding)
    Each embedder is tested with the best CE (GTE-Reranker-ModernBERT-Base without augmentation).
    Results report Stage 1 recall and final MAP side by side so you can see where the embedder makes a difference.
    Outputs: results/embedder_search/embedder_results.csv

    Usage
    -----
      python run_search.py --phase 1                   # S1 grid search
      python run_search.py --phase 2                   # S2 grid search
      python run_search.py --phase 1 --submit-only     # regenerate top-10 submissions
      python run_search.py --phase 1 --top-n 5
      python run_search.py --phase 1 --ce CE_GTE_new   # fix CE model
      python run_search.py --phase 1 --dry-run         # print plan, no execution
      python run_search.py --embedder-search           # dense embedder comparison
      python run_search.py --embedder-search --dry-run
      python run_search.py --embedder-search --ce CE_GTE_new  # override CE

    Grid sizes (defaults)
    ---------------------
        - Phase 1:  3x3x2x2x3x3 = 324 combos per CE
        - Phase 2:  3x4x4 =  48 combos  (fast)
        - Embedder: 4 models x 1 param combo = 4 runs  (very fast)
    """
    parser = argparse.ArgumentParser(
        description="JOKER: Greedy hyperparameter search (S1 -> S2 + Embedder comparison)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  --phase 1            Stage 1 param search (BM25 k1/b, RRF weights, topk, query expansion)
  --phase 2            Stage 2 param search (CE blend ratio, dedup threshold, topk)
  --embedder-search    Dense embedder comparison (bge-base vs bge-large vs bge-m3 vs bge-en-icl)
                       Fixes all pipeline params, varies only the dense model.
                       Can be run independently at any point in the workflow.

Each phase reads best_config.json written by the previous phase, runs a local
eval grid, writes results to results/<phase_dir>/<phase>_results.csv, updates
best_config.json with the winner, then generates CodaBench submissions.

The embedder search writes results/embedder_search/embedder_results.csv,
prints a verdict + recommendation for DENSE_MODEL in config.py, and
auto-generates one CodaBench submission per embedder ranked by local MAP.

Examples
--------
  python run_search.py --phase 1
  python run_search.py --phase 2
  python run_search.py --phase 3
  python run_search.py --embedder-search
  python run_search.py --embedder-search --ce CE_GTE_new
  python run_search.py --embedder-search --dry-run
  python run_search.py --embedder-search --submit-only   # regen submissions only
  python run_search.py --phase 1 --dry-run
  python run_search.py --phase 1 --submit-only
  python run_search.py --phase 1 --top-n 5
  python run_search.py --phase 2 --ce CE_GTE_new
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", type=int, choices=[1, 2], help="Phase to run (1=S1 search, 2=S2 search)")
    mode.add_argument(
        "--embedder-search", action="store_true", help="Compare dense embedder candidates with all other params fixed"
    )

    parser.add_argument(
        "--submit-only", action="store_true",
        help="Skip search; regenerate CodaBench submission files from existing CSV "
             "(works with both --phase N and --embedder-search)",
    )
    parser.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help=f"(phases only) Number of top configs to submit (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--ce", type=str, default=None,
        help="Override CE model name (default: from best_config.json or first available)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the execution plan without running any experiments",
    )
    args = parser.parse_args()

    if args.embedder_search:
        if args.submit_only:
            out_dir = os.path.join(PathManager.RESULTS_DIR, EMBEDDER_SEARCH_SUBDIR)
            done_csv = os.path.join(out_dir, EMBEDDER_SEARCH_CSV)
            if not os.path.exists(done_csv):
                print(f"ERROR: {done_csv} not found - run --embedder-search first.")
                return
            df = pd.read_csv(done_csv).sort_values("MAP", ascending=False).reset_index(drop=True)
            fixed = dict(KNOWN_GOOD_CONFIG)  # always use known-good, not best_config.json
            ce_name = _resolve_ce(args.ce, fixed)
            _generate_embedder_submissions(df, ce_name, fixed, args.top_n)
        else:
            run_embedder_search(args)
    elif args.submit_only:
        _submit_only(args.phase, args)
    else:
        _run_phase(args.phase, args)


if __name__ == "__main__":
    main()

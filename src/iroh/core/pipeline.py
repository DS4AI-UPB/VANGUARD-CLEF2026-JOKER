import argparse
import os
import shutil
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from iroh.core.config import (
    CORPUS_EMB_CACHE,
    DENSE_MODEL,
    CE_CANDIDATES, JUDGE_CANDIDATES, JUDGE_BASE_MODEL,
    STAGE1_TOPK, USE_EXPANDED_BM25,
    RRF_K, RRF_BM25_WEIGHT, RRF_DENSE_WEIGHT,
    RRF_HUMOR_BOOST,
    HUMOR_PRIOR_MIN,
    BM25_K1, BM25_B, QUERY_EXPANSION_VARIANTS,
    STAGE2_TOPK, STAGE2_CE_RANK_BLEND,
    STAGE2_DEDUP_THRESHOLD, STAGE2_CE_BATCH_SIZE,
    STAGE3_TOPK, CE_WEIGHT, JUDGE_WEIGHT, JUDGE_THRESHOLD,
    ensure_dirs,
)
from iroh.core.path_manager import PathManager
from iroh.core.utils import (
    seed_everything, load_json, save_json,
    tokenize_stemmed, expand_query,
    find_best_model, load_judge_model,
    evaluate_trec, free_gpu,
    load_corpus, build_qrel_dict, load_humor_prior,
    stage_recall_record, append_results_csv,
)

JUDGE_BATCH_SIZE = 128


@dataclass
class CorpusData:
    """Holds all corpus-level data needed by the pipeline."""
    corpus_texts: list[str]
    corpus_docids: list[str]
    bm25: BM25Okapi
    embedder: SentenceTransformer
    corpus_embeddings: torch.Tensor
    humor_prior: dict[str, float]
    humor_prior_array: np.ndarray


@dataclass
class JudgeData:
    """Holds judge model objects (all None when no judge is loaded)."""
    model: object = None
    tokenizer: object = None
    yes_id: int = 0
    no_id: int = 0

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


def build_bm25_index(
        corpus: list[dict],
        corpus_texts: list[str],
        corpus_docids: list[str],
        use_expanded: bool = True,
) -> BM25Okapi:
    """Build BM25 index with tunable k1/b, optionally with rationale-expanded docs."""
    if use_expanded and os.path.exists(PathManager.RATIONALES_FILE):
        rationale_data = load_json(PathManager.RATIONALES_FILE)
        rationale_map: dict[str, str] = {
            str(item.get("docid", "")): item.get("rationale", "")
            for item in rationale_data
            if item.get("docid") and item.get("rationale")
        }
        expanded: list[str] = []
        n_expanded = 0
        for i, _doc in enumerate(corpus):
            text = corpus_texts[i]
            rationale = rationale_map.get(corpus_docids[i], "")
            if rationale:
                expanded.append(f"{text} {rationale}")
                n_expanded += 1
            else:
                expanded.append(text)
        print(f"\tBM25: expanded {n_expanded}/{len(corpus)} docs with rationales")
        tokenized = [tokenize_stemmed(t) for t in expanded]
    else:
        print("\tBM25: using plain text")
        tokenized = [tokenize_stemmed(t) for t in corpus_texts]

    print(f"\tBM25 params: k1={BM25_K1}, b={BM25_B}")
    return BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)


def load_dense_embeddings(corpus_texts: list[str]) -> tuple[SentenceTransformer, torch.Tensor]:
    """
    Load or compute dense embeddings for the corpus.

    Embeddings stay on CPU by default. When the CE model is also loaded on the same GPU,
    putting the full corpus there can OOM. For dot products we move per-query embeddings to CPU instead.

    Set JOKER_EMBED_DEVICE=cuda in the environment to force GPU placement.
    """
    embedder = SentenceTransformer(DENSE_MODEL)
    if os.path.exists(CORPUS_EMB_CACHE):
        corpus_embeddings = torch.load(CORPUS_EMB_CACHE, weights_only=True, map_location="cpu")
        print(f"\tLoaded embeddings from cache: {corpus_embeddings.shape}")
    else:
        print("\tComputing corpus embeddings (this may take a few minutes)...")
        corpus_embeddings = embedder.encode(
            corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=128,
        )
        corpus_embeddings = corpus_embeddings.cpu()
        torch.save(corpus_embeddings, CORPUS_EMB_CACHE)
        print(f"\tSaved embeddings to {CORPUS_EMB_CACHE}")

    if os.environ.get("JOKER_EMBED_DEVICE", "").lower() == "cuda" and torch.cuda.is_available():
        corpus_embeddings = corpus_embeddings.to("cuda")
        print(f"\tMoved corpus embeddings to GPU (JOKER_EMBED_DEVICE=cuda)")

    return embedder, corpus_embeddings


def _rrf_fuse(rankings: list[tuple[np.ndarray, float]], n_docs: int, k: int = RRF_K) -> np.ndarray:
    """
    Reciprocal Rank Fusion across multiple ranked lists.

    :param rankings: list of (doc_indices_sorted_best_first, weight) tuples.
                        Each array can be a partial ranking (top-N only).
                        Docs not in it contribute 0 to the score (which is correct for RRF).
    :param n_docs: size of the corpus.
    :param k:  RRF smoothing constant (larger = flatter blend).
    :return: np.ndarray of length n_docs with fused scores.
    """
    fused = np.zeros(n_docs, dtype=np.float64)
    for ranked_indices, weight in rankings:
        contribs = weight / (k + np.arange(1, len(ranked_indices) + 1))
        fused[ranked_indices] += contribs
    return fused


def _expansion_rankings(query_text: str, bm25: BM25Okapi, top_per_expansion: int) -> list[np.ndarray]:
    """
    Run BM25 once per expansion variant and return each top-N ranking.
    These are fused via RRF (NOT max-pooled), which is more robust.
    """
    expansions = expand_query(query_text)[:QUERY_EXPANSION_VARIANTS]
    rankings = []
    for eq in expansions:
        scores = bm25.get_scores(tokenize_stemmed(eq))
        top_idx = np.argpartition(scores, -top_per_expansion)[-top_per_expansion:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        rankings.append(top_idx)
    return rankings


def _stage1_hybrid_retrieval(query_text: str, query_embedding: torch.Tensor, corpus_data: CorpusData) -> np.ndarray:
    """
    Stage 1: RRF fusion of:
        (a) BM25 with query expansions
        (b) dense cosine similarity
        (c) gated humor + pun prior boosts

    Priors are gated where only docs whose prior > MIN contribute a boost.
    This prevents the 0.5 default from silently uplifting unscored docs.

    All fusion is rank-based, so different score scales cannot dominate.

    :return: np.ndarray of top-K corpus indices (best first).
    """
    n_docs = len(corpus_data.corpus_texts)
    per_system_topk = max(STAGE1_TOPK * 2, 2000)

    bm25_rankings = _expansion_rankings(query_text, corpus_data.bm25, per_system_topk)
    bm25_fused = _rrf_fuse(
        [(r, 1.0) for r in bm25_rankings], n_docs, k=RRF_K
    )
    bm25_top = np.argpartition(bm25_fused, -per_system_topk)[-per_system_topk:]
    bm25_top = bm25_top[np.argsort(bm25_fused[bm25_top])[::-1]]

    dense_scores = util.cos_sim(
        query_embedding, corpus_data.corpus_embeddings
    )[0].cpu().numpy()
    dense_top = np.argpartition(dense_scores, -per_system_topk)[-per_system_topk:]
    dense_top = dense_top[np.argsort(dense_scores[dense_top])[::-1]]

    rankings: list[tuple[np.ndarray, float]] = [
        (bm25_top, RRF_BM25_WEIGHT),
        (dense_top, RRF_DENSE_WEIGHT),
    ]

    if RRF_HUMOR_BOOST > 0:
        humor_mask = corpus_data.humor_prior_array >= HUMOR_PRIOR_MIN
        humor_idx = np.where(humor_mask)[0]
        if humor_idx.size > 0:
            humor_idx = humor_idx[
                np.argsort(corpus_data.humor_prior_array[humor_idx])[::-1]
            ]
            rankings.append((humor_idx, RRF_HUMOR_BOOST))

    fused = _rrf_fuse(rankings, n_docs, k=RRF_K)
    top1 = np.argpartition(fused, -STAGE1_TOPK)[-STAGE1_TOPK:]
    top1 = top1[np.argsort(fused[top1])[::-1]]
    return top1


def _dedup_candidates(candidate_indices: np.ndarray, corpus_embeddings: torch.Tensor, threshold: float) -> np.ndarray:
    """
    Greedy near-duplicate suppression on the Stage-1 candidates.

    Ranking from best to worst, keep a doc only if its dense cosine sim to every already-kept doc is below threshold.

    On the JOKER corpus because many jokes are paraphrases / minor variants, and reranking near-duplicates wastes
    CE budget and pollutes the top of the final list.

    :return: The surviving indices in their original ranked order.
    """
    if threshold >= 1.0 or len(candidate_indices) <= 1:
        return candidate_indices

    cand_embs = corpus_embeddings[candidate_indices]
    cand_embs = torch.nn.functional.normalize(cand_embs.float(), dim=1)

    kept_mask = np.ones(len(candidate_indices), dtype=bool)
    kept_embs: list[torch.Tensor] = []

    for i in range(len(candidate_indices)):
        if not kept_mask[i]:
            continue
        emb = cand_embs[i]
        if kept_embs:
            stacked = torch.stack(kept_embs)
            sims = (stacked @ emb).cpu().numpy()
            if (sims >= threshold).any():
                kept_mask[i] = False
                continue
        kept_embs.append(emb)

    return candidate_indices[kept_mask]


def _ce_infer_batch_size(cross_encoder: CrossEncoder) -> int:
    """
    Pick a safe Stage-2 inference batch size based on the CE's parameter count. Models:
        - MiniLM-L6 (22M) handles STAGE2_CE_BATCH_SIZE fine
        - BGE-base (278M) OOMs at the same setting and needs approx. 8x smaller batches.
    """
    try:
        n_params = sum(p.numel() for p in cross_encoder.model.parameters())
    except Exception:
        return STAGE2_CE_BATCH_SIZE
    if n_params > 200_000_000:
        return max(STAGE2_CE_BATCH_SIZE // 8, 16)
    if n_params > 50_000_000:
        return max(STAGE2_CE_BATCH_SIZE // 4, 32)
    return STAGE2_CE_BATCH_SIZE


def _stage2_rerank(
        query_text: str,
        top1: np.ndarray,
        corpus_data: CorpusData,
        cross_encoder: CrossEncoder | None,
) -> list[tuple[int, float]]:
    """
    Stage 2: deduplicate Stage-1 candidates, cross-encoder rerank,
    then blend the CE ranking with the Stage-1 ranking on a rank basis (RRF).
    Rank-based blending is scale-free, so the CE-vs-Stage-1 mix doesn't get distorted by a single outlier logit.
    """
    candidates = _dedup_candidates(
        top1, corpus_data.corpus_embeddings, STAGE2_DEDUP_THRESHOLD,
    )

    if cross_encoder is None:
        keep = candidates[:STAGE2_TOPK]
        return [(int(idx), float(len(keep) - r)) for r, idx in enumerate(keep)]

    pairs = [[query_text, corpus_data.corpus_texts[i]] for i in candidates]
    bs = _ce_infer_batch_size(cross_encoder)
    use_amp = torch.cuda.is_available()
    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            ce_raw = cross_encoder.predict(pairs, batch_size=bs)
    else:
        ce_raw = cross_encoder.predict(pairs, batch_size=bs)
    ce_raw = np.asarray(ce_raw, dtype=np.float64)

    n = len(candidates)
    ce_order = np.argsort(ce_raw)[::-1]
    stage1_order = np.arange(n)

    fused = np.zeros(n, dtype=np.float64)
    fused[ce_order] += STAGE2_CE_RANK_BLEND / (RRF_K + np.arange(1, n + 1))
    fused[stage1_order] += (1 - STAGE2_CE_RANK_BLEND) / (RRF_K + np.arange(1, n + 1))

    final_order = np.argsort(fused)[::-1][:STAGE2_TOPK]
    return [(int(candidates[i]), float(fused[i])) for i in final_order]


def _stage3_judge_fusion(
        query_text: str,
        ranked: list[tuple[int, float]],
        corpus_data: CorpusData,
        judge: JudgeData,
) -> list[dict]:
    """
    Stage 3: judge fusion + final scoring.

    When a judge is loaded, all prompts for this query are batched into groups of JUDGE_BATCH_SIZE and
    scored in a single forward pass per batch.
    """
    scored: list[dict] = []

    if not judge.is_loaded:
        for doc_idx, stage2_score in ranked:
            docid = corpus_data.corpus_docids[doc_idx]
            scored.append({"docid": docid, "score": stage2_score})
    else:
        all_prompts: list[str] = []
        system_prompt = "You are a humor and wordplay detection judge. Answer only YES or NO."
        for doc_idx, _ in ranked:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f'Query: "{query_text}"\n'
                    f'Text: "{corpus_data.corpus_texts[doc_idx]}"\n'
                    "Is this a relevant joke?"
                )},
            ]
            all_prompts.append(
                judge.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )

        all_yes_probs: list[float] = []
        judge.tokenizer.padding_side = "left"
        for b_start in range(0, len(all_prompts), JUDGE_BATCH_SIZE):
            batch = all_prompts[b_start: b_start + JUDGE_BATCH_SIZE]
            inputs = judge.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(judge.model.device)
            with torch.no_grad():
                logits = judge.model(**inputs).logits[:, -1, :]
            y_l = logits[:, judge.yes_id]
            n_l = logits[:, judge.no_id]
            mx_l = torch.maximum(y_l, n_l)
            yes_probs = torch.exp(y_l - mx_l) / (torch.exp(y_l - mx_l) + torch.exp(n_l - mx_l))
            all_yes_probs.extend(yes_probs.cpu().tolist())

        for i, (doc_idx, stage2_score) in enumerate(ranked):
            docid = corpus_data.corpus_docids[doc_idx]
            yes_prob = all_yes_probs[i]
            final = CE_WEIGHT * stage2_score + JUDGE_WEIGHT * yes_prob
            if yes_prob < JUDGE_THRESHOLD:
                final *= 0.80
            scored.append({"docid": docid, "score": final})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:STAGE3_TOPK]


def precompute_stage1(test_queries: list[dict], corpus_data: CorpusData) -> dict[str, np.ndarray]:
    """
    Pre-compute Stage 1 hybrid retrieval for every query and cache the results in a dict keyed by qid.

    Computing Stage 1 once and reusing across multiple runs (different CE / judge combinations)
    is the single biggest speed-up when running an experiment grid. Stage 1 is identical for every run.

    :return: stage1_cache: {qid: top1_indices_array}
    """
    query_texts = [q["query"] for q in test_queries]
    print(f"\tPre-computing Stage 1 for {len(query_texts)} queries...")
    query_embeddings = corpus_data.embedder.encode(
        query_texts, convert_to_tensor=True,
        batch_size=64, show_progress_bar=False,
    )
    query_embeddings = query_embeddings.to(corpus_data.corpus_embeddings.device)

    stage1_cache: dict[str, np.ndarray] = {}
    for q_idx, query_obj in enumerate(test_queries):
        qid = str(query_obj["qid"])
        query_text = query_obj["query"]
        q_emb = query_embeddings[q_idx]
        stage1_cache[qid] = _stage1_hybrid_retrieval(query_text, q_emb, corpus_data)
        if (q_idx + 1) % 50 == 0:
            print(f"\t\tStage 1: {q_idx + 1}/{len(test_queries)}...")

    print(f"\tStage 1 pre-computation done ({len(stage1_cache)} queries cached)")
    return stage1_cache


def run_pipeline(
        run_name: str,
        test_queries: list[dict],
        qrel_dict: dict[str, set[str]],
        corpus_data: CorpusData,
        cross_encoder: CrossEncoder | None,
        judge: JudgeData,
        stage1_cache: dict[str, np.ndarray] | None = None,
        ce_stage2_cache: dict[str, list[tuple[int, float]]] | None = None,
) -> tuple[list[dict], dict[int, list[float]]]:
    """
    Run the full 3-stage pipeline on a set of queries.

    Stage 1 - accepts a pre-computed stage1_cache so the same hybrid retrieval results can be reused across multiple
        runs that differ only in CE or judge. If None, Stage 1 is computed here (single-run mode, same as before).
    Stage 2 - accepts a ce_stage2_cache so CE scores computed for one run can be reused by any subsequent run that
            shares the same CE model. Pass an empty dict on the first run with a given CE.
            It will be populated in-place and can be passed unchanged to later runs.
    Stage 3 - judge inference is batched (JUDGE_BATCH_SIZE prompts per forward pass) instead of one doc at a time.

    :param stage1_cache: optional {qid -> top1 index array} from precompute_stage1.
    :param ce_stage2_cache: optional {qid -> ranked list} that is read from (cache hit) or written to (cache miss).
                                Pass a shared dict when iterating over multiple runs that use the same CE model.
    :return: (submission_data, stage_recalls)
    """
    submission_data: list[dict] = []
    stage_recalls: dict[int, list[float]] = {1: [], 2: [], 3: []}

    if stage1_cache is None:
        query_texts = [q["query"] for q in test_queries]
        print(f"\tEncoding {len(query_texts)} queries in one batch...")
        query_embeddings = corpus_data.embedder.encode(
            query_texts, convert_to_tensor=True,
            batch_size=64, show_progress_bar=False,
        )
        query_embeddings = query_embeddings.to(corpus_data.corpus_embeddings.device)
    else:
        query_embeddings = None

    if ce_stage2_cache is None:
        ce_stage2_cache = {}

    for q_idx, query_obj in enumerate(test_queries):
        qid = str(query_obj["qid"])
        query_text = query_obj["query"]
        relevant = qrel_dict.get(qid, set())

        if stage1_cache is not None:
            top1 = stage1_cache[qid]
        else:
            q_emb = query_embeddings[q_idx]
            top1 = _stage1_hybrid_retrieval(query_text, q_emb, corpus_data)

        if relevant:
            s1 = {corpus_data.corpus_docids[i] for i in top1}
            stage_recalls[1].append(len(relevant & s1) / len(relevant))

        if qid in ce_stage2_cache:
            ranked = ce_stage2_cache[qid]
        else:
            ranked = _stage2_rerank(query_text, top1, corpus_data, cross_encoder)
            ce_stage2_cache[qid] = ranked

        if relevant:
            s2 = {corpus_data.corpus_docids[i] for i, _ in ranked}
            stage_recalls[2].append(len(relevant & s2) / len(relevant))

        scored = _stage3_judge_fusion(query_text, ranked, corpus_data, judge)

        if relevant:
            s3 = {d["docid"] for d in scored}
            stage_recalls[3].append(len(relevant & s3) / len(relevant))

        for rank, item in enumerate(scored):
            submission_data.append({
                "run_id": run_name,
                "manual": 0,
                "qid": int(qid),
                "docid": str(item["docid"]),
                "rank": rank,
                "score": round(float(item["score"]), 4),
            })

        if (q_idx + 1) % 5 == 0:
            print(f"\t\t{q_idx + 1}/{len(test_queries)} queries...")

    return submission_data, stage_recalls


def _load_shared_resources(corpus_file: str, use_expanded_bm25: bool) -> CorpusData:
    """Load corpus, BM25 index, dense embeddings, and humor prior."""
    print("1. Loading corpus...")
    corpus, corpus_texts, corpus_docids = load_corpus(corpus_file)

    humor_prior, humor_prior_array = load_humor_prior(corpus_docids)
    print(f"\tHumor prior: {len(humor_prior)} documents scored")

    print("2. Building BM25 index...")
    bm25 = build_bm25_index(corpus, corpus_texts, corpus_docids, use_expanded_bm25)

    print("3. Loading dense retriever...")
    embedder, corpus_embeddings = load_dense_embeddings(corpus_texts)

    return CorpusData(
        corpus_texts=corpus_texts,
        corpus_docids=corpus_docids,
        bm25=bm25,
        embedder=embedder,
        corpus_embeddings=corpus_embeddings,
        humor_prior=humor_prior,
        humor_prior_array=humor_prior_array,
    )


def _load_judge(judge_path: str | None) -> JudgeData:
    """Load a judge model or return an empty JudgeData."""
    if not judge_path or not os.path.exists(judge_path):
        if judge_path:
            print(f"\tWARNING: judge '{judge_path}' not found, running without judge")
        return JudgeData()

    model, tokenizer, yes_id, no_id = load_judge_model(judge_path, JUDGE_BASE_MODEL)
    return JudgeData(model=model, tokenizer=tokenizer, yes_id=yes_id, no_id=no_id)


def _package_best_submission(
        best_path: str,
        team_run_name: str,
        ablation_map: float,
        total_records: int,
        out_dir: str,
        top_n: int,
) -> None:
    """Copy the best submission to BASE and create prediction.zip."""
    submission_filename = f"{team_run_name}.json"
    dst_json = os.path.join(PathManager.BASE, submission_filename)
    dst_zip = os.path.join(PathManager.BASE, "prediction.zip")
    shutil.copy(best_path, dst_json)

    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(dst_json, submission_filename)

    print(f"\n{'=' * 70}")
    print(f"TOP-{top_n} SUBMISSIONS COMPLETE")
    print(f"{'=' * 70}")
    print(f"\tBest config: {team_run_name}")
    print(f"\tAblation MAP: {ablation_map:.4f}")
    print(f"\t{submission_filename} -> {dst_json}")
    print(f"\tprediction.zip  -> {dst_zip}")
    print(f"\tUpload prediction.zip to CodaBench!")
    print(f"\n\tAll {total_records} submissions in: {out_dir}/")


def run_top_n_submissions(
        top_n: int = 30,
        results_subdir: str = "top_submissions",
        no_judge: bool = False,
) -> None:
    """
    Load ablation results, pick the top N by MAP, and generate a CodaBench submission for each using ALL test queries.

    If no_judge=True, Stage 3 runs without a judge for every config which is much faster,
    but ablation MAP no longer predicts ranking. Used for Stages 1+2 submissions across your top CEs.
    """
    seed_everything()
    ensure_dirs()

    ablation_csv = os.path.join(PathManager.RESULTS_DIR, "ablation", "ablation_results.csv")
    if not os.path.exists(ablation_csv):
        ablation_csv = os.path.join(PathManager.RESULTS_DIR, "ablation_results.csv")
    if not os.path.exists(ablation_csv):
        print("ERROR: No ablation_results.csv found. Run ablation.py first.")
        return

    df = pd.read_csv(ablation_csv).sort_values("MAP", ascending=False).reset_index(drop=True)
    top_n = min(top_n, len(df))
    df_top = df.head(top_n)

    print(f"\nTop {top_n} configurations from ablation (ranked by MAP):")
    print("-" * 80)
    for i, row in df_top.iterrows():
        print(f"\t{int(i) + 1:>3}. {row['run_name']:<50} MAP={row['MAP']:.4f}")
    print("-" * 80)

    print("\nLoading shared resources...")
    corpus_data = _load_shared_resources(PathManager.CORPUS_FILE, USE_EXPANDED_BM25)

    test_queries = load_json(PathManager.QUERIES_TEST_FILE)
    qrel_dict: dict[str, set[str]] = {}
    print(f"\tOfficial test queries: {len(test_queries)}")

    out_dir = os.path.join(PathManager.RESULTS_DIR, results_subdir)
    os.makedirs(out_dir, exist_ok=True)

    tracker_csv = os.path.join(out_dir, "submission_tracker.csv")
    done_runs: set[str] = set()
    all_records: list[dict] = []
    if os.path.exists(tracker_csv):
        tracker_df = pd.read_csv(tracker_csv)
        done_runs = set(tracker_df["run_name"].tolist())
        all_records = tracker_df.to_dict("records")
        print(f"\t{len(done_runs)} already completed, skipping")

    runs_by_judge: dict[str, list[dict]] = defaultdict(list)
    if no_judge:
        best_per_ce: dict[str, dict] = {}
        for _, row in df_top.iterrows():
            ce_name = str(row.get("ce", "none"))
            row_d = row.to_dict()
            row_d["judge"] = "NoJudge"
            row_d["run_name"] = f"{ce_name}__NoJudge"
            if row_d["run_name"] in done_runs:
                continue
            kept = best_per_ce.get(ce_name)
            if kept is None or row_d["MAP"] > kept["MAP"]:
                best_per_ce[ce_name] = row_d
        for row_d in best_per_ce.values():
            runs_by_judge["NoJudge"].append(row_d)
        print(f"\t--no-judge: {len(best_per_ce)} unique CEs to submit")
    else:
        for _, row in df_top.iterrows():
            if row["run_name"] in done_runs:
                continue
            judge_key = str(row.get("judge", "none"))
            runs_by_judge[judge_key].append(row.to_dict())

    total_remaining = sum(len(v) for v in runs_by_judge.values())
    if total_remaining == 0:
        print("All submissions already done!")
    else:
        print(f"\n{total_remaining} submissions to generate...\n")

    print("\nPre-computing Stage 1 (computed once, reused for every run)...")
    stage1_cache = precompute_stage1(test_queries, corpus_data)
    import gc
    del corpus_data.embedder
    gc.collect()
    torch.cuda.empty_cache()
    ce_stage2_caches: dict[str, dict] = {}

    completed = 0
    for judge_key, rows in runs_by_judge.items():
        if no_judge:
            judge = JudgeData()
        elif judge_key not in ("none", "None", ""):
            judge_path = os.path.join(PathManager.MODELS_DIR, judge_key)
            print(f"Loading judge: {judge_key}...")
            judge = _load_judge(judge_path)
        else:
            judge = JudgeData()

        for row in rows:
            ce_name = str(row.get("ce", "none"))
            run_name = f"VANGUARD_Task1_{ce_name}__{judge_key}"

            print(f"\n{'-' * 60}")
            print(f"\tSUBMISSION {completed + 1}/{total_remaining}: {run_name}")
            print(f"\tCE: {ce_name} | Judge: {judge_key}")
            print(f"{'-' * 60}")

            cross_encoder: CrossEncoder | None = None
            if ce_name not in ("none", "None", ""):
                ce_path = os.path.join(PathManager.MODELS_DIR, ce_name)
                if os.path.exists(ce_path):
                    cross_encoder = CrossEncoder(ce_path)
                else:
                    print(f"\tWARNING: CE '{ce_name}' not found, running without CE")

            if ce_name not in ce_stage2_caches:
                ce_stage2_caches[ce_name] = {}
                if cross_encoder is not None:
                    print(f"\tStage-2 cache: new (first run with CE '{ce_name}')")
            else:
                print(f"\tStage-2 cache: reusing {len(ce_stage2_caches[ce_name])} "
                      f"cached queries for CE '{ce_name}'")

            start_time = time.time()
            submission_data, _ = run_pipeline(
                run_name=run_name,
                test_queries=test_queries,
                qrel_dict=qrel_dict,
                corpus_data=corpus_data,
                cross_encoder=cross_encoder,
                judge=judge,
                stage1_cache=stage1_cache,
                ce_stage2_cache=ce_stage2_caches[ce_name],
            )
            elapsed = time.time() - start_time

            sub_path = os.path.join(out_dir, f"prediction_{run_name}.json")
            save_json(submission_data, sub_path)
            n_queries = len({item["qid"] for item in submission_data})
            print(f"\tSaved: {sub_path} ({len(submission_data)} records, {n_queries} queries)")

            record = {
                "run_name": run_name,
                "ce": ce_name,
                "judge": judge_key,
                "ablation_MAP": round(row["MAP"], 4),
                "num_queries": n_queries,
                "num_records": len(submission_data),
                "time_sec": round(elapsed, 1),
            }
            all_records.append(record)

            pd.DataFrame(all_records).to_csv(tracker_csv, index=False)

            if cross_encoder is not None:
                del cross_encoder
                torch.cuda.empty_cache()

            completed += 1

        if judge.is_loaded:
            free_gpu(judge.model)

    best_row = df_top.iloc[0]
    best_ce = str(best_row.get("ce", "none"))
    best_judge = "NoJudge" if no_judge else str(best_row.get("judge", "none"))
    best_team_run = f"VANGUARD_Task1_{best_ce}__{best_judge}"
    best_path = os.path.join(out_dir, f"prediction_{best_team_run}.json")

    if os.path.exists(best_path):
        _package_best_submission(
            best_path=best_path,
            team_run_name=best_team_run,
            ablation_map=float(df_top.iloc[0]["MAP"]),
            total_records=len(all_records),
            out_dir=out_dir,
            top_n=top_n,
        )


def main() -> None:
    """
    Stage 1: Hybrid retrieval (BM25 + dense + humor prior)
    Stage 2: Cross-encoder reranking
    Stage 3: Judge fusion + final scoring

    Supports local evaluation and CodaBench submission modes.
    Saves each run immediately after completion.

    Usage:
        python pipeline.py                           # local eval (test split)
        python pipeline.py --submission              # CodaBench submission (all queries)
        python pipeline.py --ce CE_L6_earlystop --judge Judge_7B_r64_earlystop
    """
    parser = argparse.ArgumentParser(description="Run JOKER retrieval pipeline")
    parser.add_argument(
        "--submission", action="store_true", help="Single-run submission on all test queries (best CE + Judge)"
    )
    parser.add_argument(
        "--top-n", type=int, default=None, help="Generate submissions for top N ablation configs (e.g. --top-n 30)"
    )
    parser.add_argument("--ce", type=str, default=None, help="Cross-encoder model name (under MODELS_DIR)")
    parser.add_argument("--judge", type=str, default=None, help="Judge model name (under MODELS_DIR)")
    parser.add_argument(
        "--results-subdir", type=str, default="boosted", help="Subdirectory under results/ for this run"
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip Stage 3 judge entirely (Stages 1+2 only). Overrides --judge if both are set."
    )
    args = parser.parse_args()

    if args.top_n is not None:
        run_top_n_submissions(
            top_n=args.top_n,
            results_subdir=(args.results_subdir if args.results_subdir != "boosted" else "top_submissions"),
            no_judge=args.no_judge,
        )
        return

    seed_everything()
    ensure_dirs()

    results_dir = os.path.join(PathManager.RESULTS_DIR, args.results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    corpus_data = _load_shared_resources(PathManager.CORPUS_FILE, USE_EXPANDED_BM25)

    if args.submission:
        test_queries = load_json(PathManager.QUERIES_TEST_FILE)
        qrel_dict: dict[str, set[str]] = {}
        print(f"\tSubmission mode: {len(test_queries)} queries (from QUERIES_TEST_FILE)")
    else:
        test_queries = load_json(PathManager.LOCAL_TEST_QUERIES)
        qrels_data = load_json(PathManager.LOCAL_TEST_QRELS)
        qrel_dict = build_qrel_dict(qrels_data)
        print(f"\tLocal eval: {len(test_queries)} queries, {len(qrels_data)} qrels")

    if args.ce is None:
        ce_path = find_best_model(PathManager.MODELS_DIR, CE_CANDIDATES)
    else:
        ce_path = os.path.join(PathManager.MODELS_DIR, args.ce)
        if not os.path.exists(ce_path):
            print(f"ERROR: CE not found at {ce_path}")
            return

    if args.no_judge:
        judge_path = None
        print("\t--no-judge: Stage 3 will run without a judge")
    elif args.judge is None:
        judge_path = find_best_model(PathManager.MODELS_DIR, JUDGE_CANDIDATES)
    else:
        judge_path = os.path.join(PathManager.MODELS_DIR, args.judge)
        if not os.path.exists(judge_path):
            print(f"ERROR: Judge not found at {judge_path}")
            return

    TEAM_NAME = "VANGUARD"
    ce_label = os.path.basename(ce_path) if ce_path else "none"
    judge_label = os.path.basename(judge_path) if judge_path else "none"
    run_name = f"{TEAM_NAME}_Task1_{ce_label}__{judge_label}"

    print(f"\n4. Running pipeline: {run_name}")
    print(f"\t\tCE: {ce_label}")
    print(f"\t\tJudge: {judge_label}")

    cross_encoder = CrossEncoder(ce_path) if ce_path else None
    judge = _load_judge(judge_path)

    start_time = time.time()
    submission_data, stage_recalls = run_pipeline(
        run_name=run_name,
        test_queries=test_queries,
        qrel_dict=qrel_dict,
        corpus_data=corpus_data,
        cross_encoder=cross_encoder,
        judge=judge,
    )
    elapsed = time.time() - start_time

    sub_path = os.path.join(results_dir, f"submission_{run_name}.json")
    save_json(submission_data, sub_path)

    if not args.submission and qrel_dict:
        eval_metrics = evaluate_trec(submission_data, qrels_data)
    else:
        eval_metrics = {m: 0.0 for m in [
            "map", "recall_30", "recall_1000", "ndcg_cut_10", "P_5", "P_10",
        ]}

    result = {
        "run_name": run_name,
        "ce": ce_label,
        "judge": judge_label,
        "num_queries": len(test_queries),
        "num_records": len(submission_data),
        "MAP": round(eval_metrics.get("map", 0), 4),
        "R@30": round(eval_metrics.get("recall_30", 0), 4),
        "R@1000": round(eval_metrics.get("recall_1000", 0), 4),
        "NDCG@10": round(eval_metrics.get("ndcg_cut_10", 0), 4),
        "P@5": round(eval_metrics.get("P_5", 0), 4),
        "P@10": round(eval_metrics.get("P_10", 0), 4),
        **stage_recall_record(stage_recalls),
        "time_sec": round(elapsed, 1),
    }

    append_results_csv(os.path.join(results_dir, "pipeline_results.csv"), result)

    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"\tQueries: {len(test_queries)}, Records: {len(submission_data)}")
    print(f"\tTime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    if not args.submission:
        print(
            f"\tMAP={result['MAP']:.4f}  NDCG@10={result['NDCG@10']:.4f}  "
            f"R@30={result['R@30']:.4f}  R@1000={result['R@1000']:.4f}"
        )
        print(
            f"\tStage recalls: S1={result['s1_recall']:.3f}  "
            f"S2={result['s2_recall']:.3f}  S3={result['s3_recall']:.3f}"
        )
    else:
        submission_filename = f"{run_name}.json"
        dst = os.path.join(PathManager.BASE, submission_filename)
        dst_zip = os.path.join(PathManager.BASE, "prediction.zip")
        shutil.copy(sub_path, dst)

        with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(dst, submission_filename)

        print(f"\n\tSaved to: {dst}")
        print(f"\tZipped to: {dst_zip}")
        print(f"\tUpload prediction.zip to CodaBench!")

    if cross_encoder is not None:
        del cross_encoder
    if judge.is_loaded:
        free_gpu(judge.model)


if __name__ == "__main__":
    main()

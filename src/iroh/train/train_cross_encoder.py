import argparse
import gc
import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from scipy.stats import spearmanr
from sentence_transformers import CrossEncoder, InputExample
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

from iroh.core.config import CE_BASE_MODEL, CE_TRAIN_CONFIGS, ensure_dirs
from iroh.core.path_manager import PathManager
from iroh.core.utils import seed_everything, load_json, load_data_for_config, check_training_status, mark_early_stopped


def get_query_positives(data: list[dict]) -> dict[str, list[str]]:
    """Extracts text examples with a label >= 0.5, grouped by their query."""
    query_positives: dict[str, list[str]] = {}
    for item in data:
        query = item.get("query", "General Wordplay")
        text = item.get("text") or ""
        label = float(item.get("label", 0.0))
        if label >= 0.5:
            query_positives.setdefault(query, []).append(text)

    return query_positives


def tok(text: str, stemmer: PorterStemmer) -> list:
    if not text:
        return []
    return [stemmer.stem(t) for t in re.findall(r"\b\w+(?:'\w+)?\b", text.lower())]


def build_training_pairs(
        data: list[dict],
        corpus_texts: list[str],
        bm25_hard_negs: int = 5,
        corpus_negs_per_pos: int = 2,
) -> list[InputExample]:
    stemmer = PorterStemmer()

    query_positives = get_query_positives(data)

    examples = []
    for item in data:
        query = item.get("query", "General Wordplay")
        text = item.get("text") or ""
        label = float(item.get("label", 0.0))
        examples.append(InputExample(texts=[query, text], label=label))

    all_positive_texts = set()
    for texts in query_positives.values():
        all_positive_texts.update(texts)

    for query, pos_texts in query_positives.items():
        n_negs = min(len(pos_texts) * corpus_negs_per_pos, 10)
        candidates = [t for t in corpus_texts if t and t not in all_positive_texts and len(t.strip()) > 20]
        for neg in random.sample(candidates, min(n_negs, len(candidates))):
            examples.append(InputExample(texts=[query, neg], label=0.0))

    if bm25_hard_negs > 0:
        try:
            tokenized = [tok(t, stemmer) for t in corpus_texts]
            bm25 = BM25Okapi(tokenized)
            for query, pos_texts in query_positives.items():
                pos_set = set(pos_texts)
                scores = bm25.get_scores(tok(query, stemmer))
                top_idx = np.argsort(scores)[::-1][:80]
                count = 0
                for idx in top_idx:
                    if count >= bm25_hard_negs:
                        break
                    candidate = corpus_texts[idx]
                    if candidate not in pos_set and len(candidate.strip()) > 20:
                        examples.append(InputExample(texts=[query, candidate], label=0.0))
                        count += 1
            print(f"\tBM25 hard negatives added")
        except Exception as e:
            print(f"\tBM25 hard negatives skipped: {e}")

    random.shuffle(examples)
    return examples


def _per_query_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Standard MAP-style average precision for a single query.

    Sorts candidates by score (desc), walks the ranking, accumulates
    precision@k whenever a positive is hit. Returns mean precision over
    all positives (the "average precision" of MAP). Returns 0.0 if no
    positives exist.
    """
    if len(scores) == 0 or labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    labels_ranked = labels[order]
    hits = 0
    precision_sum = 0.0
    for k, lbl in enumerate(labels_ranked, start=1):
        if lbl > 0:
            hits += 1
            precision_sum += hits / k
    return precision_sum / labels.sum()


def build_eval_pairs(
        data: list[dict],
        corpus_texts: list[str],
        corpus_negs_per_pos: int = 5,
        bm25_hard_negs: int = 5,
        eval_seed: int = 1337,
) -> tuple[list[InputExample], dict[str, list[int]]]:
    """
    Build validation pairs with controlled negatives, and return per-query groups so MAP can be computed at query level.
    Uses a separate RNG seed so val negatives don't shift when train negatives are reshuffled across epochs.

    More corpus negatives by default (5 vs 2) so each query has enough candidates for AP to be meaningful.

    :return: (examples, query_groups), query_groups maps query strings to the list of example indices belonging to it.
    """
    stemmer = PorterStemmer()
    rng = random.Random(eval_seed)

    query_positives = get_query_positives(data)

    examples: list[InputExample] = []
    query_groups: dict[str, list[int]] = {}

    for item in data:
        query = item.get("query", "General Wordplay")
        text = item.get("text") or ""
        label = float(item.get("label", 0.0))
        idx = len(examples)
        examples.append(InputExample(texts=[query, text], label=label))
        query_groups.setdefault(query, []).append(idx)

    all_positive_texts = set()
    for texts in query_positives.values():
        all_positive_texts.update(texts)

    for query, pos_texts in query_positives.items():
        n_negs = min(len(pos_texts) * corpus_negs_per_pos, 20)
        candidates = [
            t for t in corpus_texts
            if t and t not in all_positive_texts and len(t.strip()) > 20
        ]
        if not candidates:
            continue
        for neg in rng.sample(candidates, min(n_negs, len(candidates))):
            idx = len(examples)
            examples.append(InputExample(texts=[query, neg], label=0.0))
            query_groups.setdefault(query, []).append(idx)

    if bm25_hard_negs > 0:
        try:
            tokenized = [tok(t, stemmer) for t in corpus_texts]
            bm25 = BM25Okapi(tokenized)
            for query, pos_texts in query_positives.items():
                pos_set = set(pos_texts)
                scores = bm25.get_scores(tok(query, stemmer))
                top_idx = np.argsort(scores)[::-1][:120]
                count = 0
                for cand_i in top_idx:
                    if count >= bm25_hard_negs:
                        break
                    candidate = corpus_texts[cand_i]
                    if candidate not in pos_set and len(candidate.strip()) > 20:
                        idx = len(examples)
                        examples.append(InputExample(texts=[query, candidate], label=0.0))
                        query_groups.setdefault(query, []).append(idx)
                        count += 1
        except Exception as e:
            print(f"\tVal BM25 hard negatives skipped: {e}")

    return examples, query_groups


def evaluate_ce(
        model: CrossEncoder,
        val_examples: list[InputExample],
        val_query_groups: dict[str, list[int]] | None = None,
) -> dict:
    """
    Evaluate a cross-encoder on the validation set.

    If val_query_groups is provided (dict: query to list of example indices), MAP is computed per query then averaged.
    Otherwise falls back to a single global ranking (less informative but always works).

    :return: A dict with:
              - val_map:    mean per-query average precision (PRIMARY metric which is what CodaBench actually measures,
                            and it's used for early stopping)
              - val_auc:    threshold-free AUC across all pairs (robust to class imbalance)
              - score_gap:  mean(positive scores) - mean(negative scores)
              - val_mse:    MSE in logit space (kept for backward compatibility, but DO NOT use for early stopping)
              - n_pos / n_neg / n_queries: composition for debugging
    """
    val_pairs = [ex.texts for ex in val_examples]
    val_labels = np.array([ex.label for ex in val_examples], dtype=np.float64)
    val_preds = np.asarray(model.predict(val_pairs, batch_size=256), dtype=np.float64)

    label_bin = (val_labels >= 0.5).astype(int)
    n_pos = int(label_bin.sum())
    n_neg = int(len(label_bin) - n_pos)

    if val_query_groups:
        per_query_aps = []
        for _query, idxs in val_query_groups.items():
            if len(idxs) < 2:
                continue
            idx_arr = np.asarray(idxs, dtype=int)
            q_labels = label_bin[idx_arr]
            if q_labels.sum() == 0:
                continue
            q_scores = val_preds[idx_arr]
            per_query_aps.append(_per_query_average_precision(q_scores, q_labels))
        val_map = float(np.mean(per_query_aps)) if per_query_aps else 0.0
        n_queries = len(per_query_aps)
    else:
        val_map = _per_query_average_precision(val_preds, label_bin)
        n_queries = 1

    if n_pos > 0 and n_neg > 0:
        try:
            val_auc = float(roc_auc_score(label_bin, val_preds))
        except Exception:
            val_auc = 0.0
    else:
        val_auc = 0.0

    pos_scores = val_preds[label_bin == 1]
    neg_scores = val_preds[label_bin == 0]
    gap = (float(pos_scores.mean() - neg_scores.mean()) if n_pos > 0 and n_neg > 0 else 0.0)

    mse = float(mean_squared_error(val_labels, val_preds))
    try:
        sp, _ = spearmanr(val_labels, val_preds)
        sp = float(sp) if np.isfinite(sp) else 0.0
    except Exception:
        sp = 0.0

    return {
        "val_map": val_map,
        "val_auc": val_auc,
        "score_gap": gap,
        "mse": mse,
        "spearman": sp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_queries": n_queries,
    }


def train_one_config(
        config: dict,
        corpus_texts: list[str],
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {config['name']}")
    print(f"\tLR: {config['lr']}, Max epochs: {config['max_epochs']}, Patience: {config['patience']}")
    print(f"{'=' * 60}")

    original_data, aug_data = load_data_for_config(
        config,
        fallback_rationale=PathManager.RATIONALES_FILE,
        fallback_augmented=PathManager.AUGMENTED_FILE,
    )

    if not original_data:
        print(f"\tSKIPPING - no training data found")
        return {"experiment": config["name"], "error": "no data"}
    split_seed = abs(hash(config["name"])) % (2 ** 31)
    split_rng = random.Random(split_seed)

    positives = [d for d in original_data if float(d.get("label", 0.0)) >= 0.5]
    negatives = [d for d in original_data if float(d.get("label", 0.0)) < 0.5]
    split_rng.shuffle(positives)
    split_rng.shuffle(negatives)
    print(f"\tSource data: {len(positives)} positives, {len(negatives)} negatives (split_seed={split_seed})")

    n_pos_val = max(int(len(positives) * 0.1), 3 if len(positives) >= 6 else 1)
    n_neg_val = max(int(len(negatives) * 0.1), 3 if len(negatives) >= 6 else 0)

    val_data = positives[:n_pos_val] + negatives[:n_neg_val]
    train_data = positives[n_pos_val:] + negatives[n_neg_val:]
    split_rng.shuffle(val_data)
    split_rng.shuffle(train_data)

    train_pos = sum(1 for d in train_data if float(d.get("label", 0.0)) >= 0.5)
    if train_pos < 5:
        print(f"\t! WARNING: only {train_pos} positives in train set. "
              f"Model will likely degenerate to 'always negative'. "
              f"Consider enabling use_augmented=True or merging more data.")
    if len(val_data) < 6:
        print(f"\t! WARNING: only {len(val_data)} val items - early-stopping signal will be noisy.")

    train_examples_original = build_training_pairs(train_data, corpus_texts)

    val_examples, val_query_groups = build_eval_pairs(
        val_data, corpus_texts,
        corpus_negs_per_pos=5,
        bm25_hard_negs=5,
        eval_seed=1337,
    )

    augmented_examples = []
    if config["use_augmented"] and aug_data:
        augmented_examples = build_training_pairs(aug_data, corpus_texts)
        print(f"\tAugmented pairs: {len(augmented_examples)}")

    if config["use_augmented"]:
        train_examples = train_examples_original + augmented_examples
        random.shuffle(train_examples)
        print(f"\tTrain pairs: {len(train_examples)} (original + augmented)")
    else:
        train_examples = train_examples_original[:]
        print(f"\tTrain pairs: {len(train_examples)} (original only)")

    n_val_pos = sum(1 for ex in val_examples if ex.label >= 0.5)
    n_val_neg = len(val_examples) - n_val_pos
    print(f"\tVal pairs: {len(val_examples)}  "
          f"({n_val_pos} pos / {n_val_neg} neg across {len(val_query_groups)} queries)")

    train_dl = DataLoader(train_examples, shuffle=True, batch_size=config["batch_size"])

    base_model_id = config.get("base_model", CE_BASE_MODEL)
    automodel_args = config.get("automodel_args") or {}
    print(f"\tBase model: {base_model_id}" + (f"\tautomodel_args={automodel_args}" if automodel_args else ""))
    model = CrossEncoder(base_model_id, num_labels=1, automodel_args=automodel_args if automodel_args else None)
    warmup = math.ceil(len(train_dl) * config["max_epochs"] * config["warmup_ratio"])

    save_dir = os.path.join(PathManager.MODELS_DIR, config["name"])
    os.makedirs(save_dir, exist_ok=True)

    best_val_map = -1.0
    best_epoch_metrics: dict = {}
    patience_counter = 0
    best_epoch = 0
    epoch_history = []
    start_time = time.time()

    for epoch in range(1, config["max_epochs"] + 1):
        model.fit(
            train_dataloader=train_dl, evaluator=None, epochs=1,
            warmup_steps=warmup if epoch == 1 else 0,
            output_path=None, show_progress_bar=True,
            optimizer_params={"lr": config["lr"]},
            weight_decay=config["weight_decay"],
        )

        metrics = evaluate_ce(model, val_examples, val_query_groups)
        epoch_history.append({"epoch": epoch, **metrics})

        improved = "  << NEW BEST" if metrics["val_map"] > best_val_map else ""
        print(f"\tEpoch {epoch}: val_MAP={metrics['val_map']:.4f}  "
              f"AUC={metrics['val_auc']:.4f}  gap={metrics['score_gap']:.4f}  "
              f"mse={metrics['mse']:.3f}{improved}")

        if metrics["val_map"] > best_val_map:
            best_val_map = metrics["val_map"]
            best_epoch = epoch
            best_epoch_metrics = {"epoch": epoch, **metrics}
            patience_counter = 0
            model.save(save_dir)
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"\tEARLY STOP at epoch {epoch} (best was epoch {best_epoch}, val_MAP={best_val_map:.4f})")
            mark_early_stopped(save_dir)
            break

    elapsed = time.time() - start_time
    bm = best_epoch_metrics

    result = {
        "experiment": config["name"],
        "base_model": base_model_id,
        "data_variant": os.path.basename(config.get("rationale_file", "default")),
        "use_augmented": config["use_augmented"],
        "best_epoch": best_epoch,
        "total_epochs": len(epoch_history),
        "best_val_map": round(bm.get("val_map", 0.0), 4),
        "best_val_auc": round(bm.get("val_auc", 0.0), 4),
        "best_score_gap": round(bm.get("score_gap", 0.0), 4),
        "best_val_mse": round(bm.get("mse", 0.0), 4),
        "best_val_spearman": round(bm.get("spearman", 0.0), 4),
        "n_val_pos": bm.get("n_pos", 0),
        "n_val_neg": bm.get("n_neg", 0),
        "n_val_queries": bm.get("n_queries", 0),
        "train_pairs": len(train_examples),
        "val_pairs": len(val_examples),
        "split_seed": split_seed,
        "time_sec": round(elapsed, 1),
    }

    pd.DataFrame(epoch_history).to_csv(
        os.path.join(PathManager.RESULTS_DIR, f"ce_{config['name']}_history.csv"), index=False
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    """
    Train cross-encoder reranking models with early stopping and clean validation.
    Each config specifies its own data files (new/old/combined/base).

    Usage:
        - python train_cross_encoder.py                    # train all configs
        - python train_cross_encoder.py --config 0         # train only first config
    """
    parser = argparse.ArgumentParser(description="Train JOKER cross-encoder models")
    parser.add_argument(
        "--config", type=int, default=None, help="Train only config at this index (0-based)"
    )
    args = parser.parse_args()

    seed_everything()
    ensure_dirs()

    corpus = load_json(PathManager.CORPUS_FILE)
    corpus_texts = [doc.get("text", "") for doc in corpus]

    configs = CE_TRAIN_CONFIGS
    if args.config is not None:
        configs = [configs[args.config]]

    experiments_csv = os.path.join(PathManager.RESULTS_DIR, "ce_experiments.csv")
    done_experiments: set[str] = set()
    all_metrics: list[dict] = []
    if os.path.exists(experiments_csv):
        existing_df = pd.read_csv(experiments_csv)
        all_metrics = existing_df.to_dict("records")
        done_experiments = set(existing_df["experiment"].tolist())
        print(f"\tLoaded {len(done_experiments)} previous results from {experiments_csv}")

    for config in configs:
        save_dir = os.path.join(PathManager.MODELS_DIR, config["name"])
        status, _ = check_training_status(save_dir)
        if status == "skip":
            print(f"-\tSkipping '{config['name']}' - already completed or early stopped.")
            continue
        if config["name"] in done_experiments:
            print(f"-\tSkipping '{config['name']}' - already in results CSV.")
            continue

        result = train_one_config(config, corpus_texts)
        all_metrics.append(result)

        pd.DataFrame(all_metrics).to_csv(experiments_csv, index=False)

    best_csv = os.path.join(PathManager.RESULTS_DIR, "ce_best_results.csv")
    valid = [m for m in all_metrics if "error" not in m]
    if valid:
        best_df = pd.DataFrame(valid).sort_values("best_val_map", ascending=False).reset_index(drop=True)
        best_df.insert(0, "rank", range(1, len(best_df) + 1))
        best_df.to_csv(best_csv, index=False)
        print(f"\nBest-results summary saved to: {best_csv}")

    print(f"\n{'=' * 60}")
    print("Cross-Encoder Training Complete")
    print(f"{'=' * 60}")
    for m in sorted(all_metrics, key=lambda x: x.get("best_val_map", 0.0), reverse=True):
        if "error" in m:
            print(f"\t{m['experiment']:<40} SKIPPED ({m['error']})")
        else:
            print(f"\t{m['experiment']:<40} "
                  f"best_ep={m['best_epoch']:>2}/{m['total_epochs']:<2}  "
                  f"MAP={m['best_val_map']:.4f}  "
                  f"AUC={m['best_val_auc']:.4f}  "
                  f"Gap={m['best_score_gap']:.4f}  "
                  f"(pos/neg={m['n_val_pos']}/{m['n_val_neg']})")


if __name__ == "__main__":
    main()

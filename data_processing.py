import argparse
import os
import random
import re
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ensure_dirs
from path_manager import PathManager
from utils import seed_everything, load_json, save_json

LOCAL_TRAIN_QRELS = os.path.join(PathManager.DATA_DIR, "local_train_qrels.json")


def load_and_split_corpus():
    """
    Load raw JOKER data, create balanced train set, and local eval split.

    :return: (jokes, sampled_non_jokes, train_queries, test_queries, test_qrels)
    """
    corpus = load_json(PathManager.CORPUS_FILE)
    qrels = load_json(PathManager.QRELS_TRAIN_FILE)
    queries = load_json(PathManager.QUERIES_TRAIN_FILE)

    positive_doc_ids = {str(item["docid"]) for item in qrels if item.get("qrel") == 1}

    jokes, non_jokes = [], []
    for doc in corpus:
        doc_id = str(doc["docid"])
        text = doc.get("text", "")
        if not text or len(text.strip()) < 5:
            continue
        entry = {"text": text, "label": 1 if doc_id in positive_doc_ids else 0, "docid": doc_id}
        (jokes if doc_id in positive_doc_ids else non_jokes).append(entry)

    print(f"Corpus: {len(corpus)}, Jokes: {len(jokes)}, Non-jokes: {len(non_jokes)}")

    sampled = random.sample(non_jokes, min(len(jokes), len(non_jokes)))
    final_data = jokes + sampled
    random.shuffle(final_data)

    save_json(final_data, PathManager.PROCESSED_TRAIN_FILE)
    print(f"Balanced dataset: {len(final_data)} ({len(jokes)} + {len(sampled)})")

    random.shuffle(queries)
    split_idx = int(len(queries) * 0.8)
    train_queries = queries[:split_idx]
    test_queries = queries[split_idx:]
    test_qids = {str(q["qid"]) for q in test_queries}

    train_qrels = [q for q in qrels if str(q["qid"]) not in test_qids]
    test_qrels = [q for q in qrels if str(q["qid"]) in test_qids]

    save_json(train_queries, PathManager.LOCAL_TRAIN_QUERIES)
    save_json(test_queries, PathManager.LOCAL_TEST_QUERIES)
    save_json(train_qrels, LOCAL_TRAIN_QRELS)
    save_json(test_qrels, PathManager.LOCAL_TEST_QRELS)

    print(f"Train queries: {len(train_queries)}, Test queries: {len(test_queries)}")
    print(f"Train qrels: {len(train_qrels)}, Test qrels: {len(test_qrels)}")

    return jokes, sampled, train_queries, test_queries, test_qrels


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add linguistic features to a DataFrame with a 'text' column."""
    df = df.copy()
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["text"].str.len()
    df["avg_word_len"] = df["char_count"] / df["word_count"].clip(lower=1)
    df["punc_count"] = df["text"].apply(lambda x: len(re.findall(r"[^\w\s]", str(x))))
    df["punc_density"] = df["punc_count"] / df["word_count"].clip(lower=1)
    df["has_quotes"] = df["text"].str.contains(r"[''\"\"'']", regex=True).astype(int)
    df["has_ellipsis"] = df["text"].str.contains(r"\.{2,}|…", regex=True).astype(int)
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["uppercase_ratio"] = df["text"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    df["is_tom_swifty"] = df["text"].str.contains(
        r"said Tom|Tom\s+\w+ly|''.*said\s+Tom", regex=True, case=False
    ).astype(int)
    return df


def generate_plots(df: pd.DataFrame, qrels_per_query: Counter, jokes_n: int, non_jokes_n: int):
    """Generate publication-quality analysis plots."""
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    COLORS = {"Non-Joke": "#3B82F6", "Joke": "#EF4444"}
    df["label_str"] = df["label"].map({0: "Non-Joke", 1: "Joke"})

    _, ax = plt.subplots(figsize=(8, 4.5))
    for label, color in COLORS.items():
        subset = df[df["label_str"] == label]["word_count"]
        ax.hist(subset, bins=40, alpha=0.6, color=color, label=label, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Words per Document")
    ax.set_ylabel("Frequency")
    ax.set_title("Word Count Distribution by Label")
    ax.legend(frameon=False)
    ax.set_xlim(0, df["word_count"].quantile(0.98))
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(PathManager.PLOT_DIR, f"01_word_count_distribution.{ext}"))
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, feature, title in zip(
            axes,
            ["word_count", "punc_density", "avg_word_len"],
            ["Word Count", "Punctuation Density", "Avg Word Length"],
    ):
        parts = ax.violinplot(
            [df[df["label_str"] == "Non-Joke"][feature].values,
             df[df["label_str"] == "Joke"][feature].values],
            positions=[0, 1], showmeans=True, showmedians=True,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(list(COLORS.values())[i])
            pc.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-Joke", "Joke"])
        ax.set_title(title)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(PathManager.PLOT_DIR, f"02_feature_comparison_violin.{ext}"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].pie(
        [jokes_n, non_jokes_n], labels=["Jokes", "Non-Jokes"],
        colors=list(COLORS.values()), autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 11},
    )
    axes[0].set_title("Dataset Class Balance")
    if qrels_per_query:
        counts = sorted(qrels_per_query.values())
        axes[1].hist(counts, bins=20, color="#6366F1", edgecolor="white", linewidth=0.5)
        axes[1].set_xlabel("Relevant Documents per Query")
        axes[1].set_ylabel("Number of Queries")
        axes[1].set_title("Query Difficulty Distribution")
        axes[1].axvline(np.mean(counts), color="#EF4444", linestyle="--",
                        label=f"Mean: {np.mean(counts):.1f}")
        axes[1].legend(frameon=False)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(PathManager.PLOT_DIR, f"03_corpus_composition.{ext}"))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_cols = ["word_count", "avg_word_len", "punc_density", "has_quotes",
                 "exclamation_count", "question_count", "uppercase_ratio", "label"]
    corr_matrix = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix")
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(PathManager.PLOT_DIR, f"04_correlation_heatmap.{ext}"))
    plt.close()

    print(f"All plots saved to {PathManager.PLOT_DIR}/")


def main():
    """
    Corpus loading, balanced sampling, train/test splitting, and analysis.

    Usage:
        - python data_processing.py              # process data
        - python data_processing.py --plots      # also generate analysis plots
    """
    parser = argparse.ArgumentParser(description="JOKER data processing")
    parser.add_argument("--plots", action="store_true", help="Generate analysis plots")
    args = parser.parse_args()

    seed_everything()
    ensure_dirs()

    jokes, sampled, _train_q, _test_q, test_qrels = load_and_split_corpus()

    if args.plots:
        data = load_json(PathManager.PROCESSED_TRAIN_FILE)
        df = pd.DataFrame(data)
        df = extract_features(df)

        features = ["word_count", "avg_word_len", "punc_density", "has_quotes",
                    "exclamation_count", "question_count", "uppercase_ratio"]
        df["label_str"] = df["label"].map({0: "Non-Joke", 1: "Joke"})
        stats = df.groupby("label_str")[features].agg(["mean", "std", "median"]).round(3)
        stats.to_csv(os.path.join(PathManager.RESULTS_DIR, "data_statistics.csv"))
        print(stats)
        print(f"\nSplit info -> train queries: {len(_train_q)}, test queries: {len(_test_q)}")

        qrels_per_query = Counter(str(q["qid"]) for q in test_qrels)
        generate_plots(df, qrels_per_query, len(jokes), len(sampled))

    print("Data processing complete.")


if __name__ == "__main__":
    main()

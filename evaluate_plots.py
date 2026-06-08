from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ensure_dirs
from path_manager import PathManager
from utils import apply_plot_style

COL_MAP = "MAP"
COL_NDCG10 = "NDCG@10"
COL_RECALL30 = "Recall@30"
COL_P5 = "P@5"
COL_P10 = "P@10"
IR_METRICS = [COL_MAP, COL_NDCG10, COL_RECALL30, COL_P5, COL_P10]

LABEL_ACCURACY_UP = "Accuracy (↑)"
LATEX_HLINE = r"\hline"

PAL = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#DB2777", "#0891B2"]

apply_plot_style()


def save_plot(name: str) -> None:
    plt.savefig(os.path.join(PathManager.PLOT_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(PathManager.PLOT_DIR, f"{name}.png"))
    plt.close()
    print(f"\t\tDone: {name}")


def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(PathManager.RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _name_col(df: pd.DataFrame, fallback: str = "run_name") -> str:
    """Return the first available identifier column."""
    for c in [fallback, "run_name", "experiment", "name", "config"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _available(df: pd.DataFrame) -> bool:
    return not df.empty


def _best_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df[COL_MAP].idxmax()]


def _pick_first(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name present in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _add_bar_labels(ax: plt.Axes, bars: matplotlib.container.BarContainer, fmt: str = ".4f", fontsize: int = 8) -> None:
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():{fmt}}", ha="center", va="bottom", fontsize=fontsize
        )


def _first_non_empty(*dfs: pd.DataFrame) -> pd.DataFrame:
    for df in dfs:
        if _available(df):
            return df
    return pd.DataFrame()


def plot_round_comparison(df_r1: pd.DataFrame, df_r2: pd.DataFrame) -> bool:
    if not (_available(df_r1) and _available(df_r2)):
        return False

    best_r1 = _best_row(df_r1)
    best_r2 = _best_row(df_r2)

    r1_vals = [best_r1[m] for m in IR_METRICS]
    r2_vals = [best_r2[m] for m in IR_METRICS]

    _, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(IR_METRICS))
    w = 0.35

    bars1 = ax.bar(
        x - w / 2, r1_vals, w, label=f'Round 1: {best_r1.get("run_name", "best")[:25]}', color=PAL[0], alpha=0.8
    )
    bars2 = ax.bar(
        x + w / 2, r2_vals, w, label=f'Round 2: {best_r2.get("run_name", "best")[:25]}', color=PAL[2], alpha=0.8
    )

    _add_bar_labels(ax, bars1)
    _add_bar_labels(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(IR_METRICS)
    ax.set_ylabel("Score")
    ax.set_title("Round 1 vs Round 2 (Optimized): Best Run Comparison")
    ax.legend(frameon=True, framealpha=0.9, fontsize=8)
    save_plot("round1_vs_round2")
    return True


def plot_stage_recall(df_r1: pd.DataFrame, df_r2: pd.DataFrame) -> bool:
    if not (_available(df_r1) and _available(df_r2)):
        return False

    best_r1 = _best_row(df_r1)
    best_r2 = _best_row(df_r2)

    r1_stages = [best_r1.get(f"stage{i}_recall", 0) for i in range(1, 4)]
    r2_stages = [best_r2.get(f"stage{i}_recall", 0) for i in range(1, 4)]

    _, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    w = 0.35

    ax.bar(x - w / 2, r1_stages, w, label="Round 1", color=PAL[0], alpha=0.8)
    ax.bar(x + w / 2, r2_stages, w, label="Round 2 (Optimized)", color=PAL[2], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Stage 1\n(Retrieval)", "Stage 2\n(CE Reranking)", "Stage 3\n(+ Judge)"])
    ax.set_ylabel("Mean Recall")
    ax.set_title("Stage-Level Recall: Before vs After Optimization")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True)

    for i in range(3):
        diff = r2_stages[i] - r1_stages[i]
        if diff != 0:
            ax.annotate(
                f"{diff:+.3f}",
                xy=(i + w / 2, r2_stages[i]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8,
                color=PAL[2] if diff > 0 else PAL[1],
            )

    save_plot("stage_recall_improvement")
    return True


def plot_pipeline_leaderboard(df_leader: pd.DataFrame) -> bool:
    if not _available(df_leader) or COL_MAP not in df_leader.columns:
        return False

    df_sorted = df_leader.sort_values(COL_MAP, ascending=True)
    _, ax = plt.subplots(figsize=(12, max(4, len(df_sorted) * 0.6)))
    y = range(len(df_sorted))

    bars = ax.barh(y, df_sorted[COL_MAP], color=PAL[0], alpha=0.8, edgecolor="white")
    best_idx = int(df_sorted[COL_MAP].values.argmax())
    bars[best_idx].set_color(PAL[2])
    bars[best_idx].set_alpha(1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["run_name"], fontsize=8)
    ax.set_xlabel(COL_MAP)
    ax.set_title("Pipeline Leaderboard")

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ndcg = row.get(COL_NDCG10, 0)
        ax.text(row[COL_MAP] + 0.002, i, f"MAP={row[COL_MAP]:.4f}  NDCG={ndcg:.4f}", va="center", fontsize=7)

    plt.tight_layout()
    save_plot("pipeline_leaderboard")
    return True


def _plot_grouped_bars(df: pd.DataFrame, cols_spec: list[tuple[str, str]],
                       name_col: str, title: str, plot_name: str) -> bool:
    """Reusable grouped-bar chart for experiment comparison."""
    available_cols = [(c, t) for c, t in cols_spec if c in df.columns]
    if not available_cols:
        return False

    n_panels = min(len(available_cols), 3)
    _, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    nc = _name_col(df, name_col)
    for ax, (col, col_title) in zip(axes, available_cols[:n_panels]):
        bars = ax.bar(
            range(len(df)), df[col],
            color=[PAL[i % len(PAL)] for i in range(len(df))],
            edgecolor="white",
        )
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df[nc], rotation=35, ha="right", fontsize=7)
        ax.set_title(col_title)
        _add_bar_labels(ax, bars, fmt=".3f", fontsize=7)

    if title:
        plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_plot(plot_name)
    return True


def plot_ce_comparison(dfs: list[pd.DataFrame]) -> bool:
    non_empty = [df for df in dfs if _available(df)]
    if not non_empty:
        return False
    df_all = pd.concat(non_empty, ignore_index=True)
    cols_spec = [
        ("val_mse", "Validation MSE (↓)"),
        ("val_accuracy", LABEL_ACCURACY_UP),
        ("score_gap", "Pos–Neg Score Gap (↑)"),
    ]
    return _plot_grouped_bars(df_all, cols_spec, "experiment", "", "ce_all_comparison")


def plot_ce_training_curves(ce_hists: dict[str, pd.DataFrame]) -> bool:
    non_empty = {n: df for n, df in ce_hists.items() if _available(df)}
    if not non_empty:
        return False

    _, ax = plt.subplots(figsize=(8, 5))
    for i, (name, df_h) in enumerate(non_empty.items()):
        loss_col = _pick_first(df_h, "val_loss", "eval_loss", "train_loss")
        epoch_col = _pick_first(df_h, "epoch", "step")
        if loss_col is None or epoch_col is None:
            continue
        sub = df_h.sort_values(epoch_col).drop_duplicates(subset=epoch_col, keep="first")
        label = name.replace("ce_", "").replace("_history.csv", "")
        ax.plot(sub[epoch_col], sub[loss_col], "o-", color=PAL[i % len(PAL)], label=label, markersize=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Cross-Encoder Training Curves")
    ax.legend(frameon=True, fontsize=7)
    save_plot("ce_training_curves")
    return True


def plot_judge_comparison(dfs: list[pd.DataFrame]) -> bool:
    non_empty = [df for df in dfs if _available(df)]
    if not non_empty:
        return False
    df_all = pd.concat(non_empty, ignore_index=True)
    cols_spec = [
        ("eval_accuracy", LABEL_ACCURACY_UP),
        ("eval_f1", "F1 (↑)"),
        ("eval_precision", "Precision (↑)"),
        ("eval_recall", "Recall (↑)"),
        ("eval_auc_roc", "AUC-ROC (↑)"),
        ("val_accuracy", LABEL_ACCURACY_UP),
        ("val_loss", "Validation Loss (↓)"),
    ]
    return _plot_grouped_bars(
        df_all, cols_spec, "experiment", "Judge Model Comparison (All Experiments)", "judge_all_comparison"
    )


def plot_judge_curves(dfs: list[pd.DataFrame]) -> bool:
    non_empty = [df for df in dfs if _available(df)]
    if not non_empty:
        return False

    df_all = pd.concat(non_empty, ignore_index=True)
    _, ax = plt.subplots(figsize=(8, 5))

    for i, name in enumerate(df_all["experiment"].unique()):
        sub = df_all[df_all["experiment"] == name].sort_values("epoch")
        sub = sub.drop_duplicates(subset="epoch", keep="first")
        ax.plot(sub["epoch"], sub["eval_loss"], "o-", color=PAL[i % len(PAL)], label=name, markersize=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Judge Training Curves (All Experiments)")
    ax.legend(frameon=True, fontsize=7)
    save_plot("judge_curves_all")
    return True


def plot_ablation(df_ablation: pd.DataFrame) -> bool:
    if not _available(df_ablation) or COL_MAP not in df_ablation.columns:
        return False

    df_sorted = df_ablation.sort_values(COL_MAP, ascending=True)
    _, ax = plt.subplots(figsize=(12, max(4, len(df_sorted) * 0.5)))
    y = range(len(df_sorted))
    bars = ax.barh(y, df_sorted[COL_MAP], color=PAL[0], alpha=0.8)

    best_idx = int(df_sorted[COL_MAP].values.argmax())
    bars[best_idx].set_color(PAL[2])

    nc = _name_col(df_sorted)
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted[nc], fontsize=7)
    ax.set_xlabel(COL_MAP)
    ax.set_title("Ablation Study: All CE with Judge Combinations")
    plt.tight_layout()
    save_plot("ablation_leaderboard")
    return True


def plot_embedder_comparison(df_emb: pd.DataFrame) -> bool:
    """Table 3 in paper: 4 BGE-family embedders compared on MAP and S1 recall."""
    if not _available(df_emb) or "MAP" not in df_emb.columns:
        return False

    df_sorted = df_emb.sort_values("MAP", ascending=False).reset_index(drop=True)
    nc = _name_col(df_sorted, "embedder")

    _, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    colors = [PAL[0] if row[nc] != "bge-base" else PAL[2] for _, row in df_sorted.iterrows()]
    bars = ax.bar(range(len(df_sorted)), df_sorted["MAP"], color=colors, alpha=0.85, edgecolor="white")
    _add_bar_labels(ax, bars, fmt=".4f", fontsize=8)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted[nc], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("CodaBench MAP")
    ax.set_title("Dense Embedder - CodaBench MAP")

    ax = axes[1]
    recall_col = _pick_first(df_sorted, "s1_recall", "stage1_recall")
    if recall_col:
        bars2 = ax.bar(range(len(df_sorted)), df_sorted[recall_col], color=colors, alpha=0.85, edgecolor="white")
        _add_bar_labels(ax, bars2, fmt=".4f", fontsize=8)
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted[nc], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Stage 1 Recall")
        ax.set_title("Dense Embedder - Stage 1 Recall")
    else:
        axes[1].set_visible(False)

    plt.suptitle("Dense Embedder Comparison (Table 3)", fontsize=13, y=1.02)
    plt.tight_layout()
    save_plot("embedder_comparison")
    return True


def plot_dataset_stats(df_stats: pd.DataFrame) -> bool:
    if not _available(df_stats):
        return False

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    split_col = _name_col(df_stats, "split")

    count_cols = [
        c for c in ["num_queries", "num_corpus", "num_docs", "num_qrels", "total_relevant"] if c in df_stats.columns
    ]
    if count_cols:
        ax = axes[0]
        x = np.arange(len(count_cols))
        w = 0.8 / max(len(df_stats), 1)
        for i, (_, row) in enumerate(df_stats.iterrows()):
            vals = [row.get(c, 0) for c in count_cols]
            ax.bar(x + i * w, vals, w, label=row[split_col], color=PAL[i % len(PAL)], alpha=0.8)
        ax.set_xticks(x + w * (len(df_stats) - 1) / 2)
        ax.set_xticklabels([c.replace("num_", "").replace("_", " ").title() for c in count_cols], fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Dataset Size by Split")
        ax.legend(frameon=True, fontsize=8)
    else:
        axes[0].set_visible(False)

    avg_col = _pick_first(df_stats, "avg_relevant_per_query", "avg_rel_per_query", "mean_relevant")
    if avg_col is not None:
        ax = axes[1]
        bars = ax.bar(
            df_stats[split_col], df_stats[avg_col], color=[PAL[i % len(PAL)] for i in range(len(df_stats))], alpha=0.8
        )
        for bar, val in zip(bars, df_stats[avg_col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9
            )
        ax.set_ylabel("Avg Relevant Docs / Query")
        ax.set_title("Relevance Density")
    else:
        axes[1].set_visible(False)

    plt.suptitle("JOKER Task 1 - Dataset Statistics", fontsize=13, y=1.02)
    plt.tight_layout()
    save_plot("dataset_statistics")
    return True


def plot_submission(df_submission: pd.DataFrame) -> bool:
    if not _available(df_submission):
        return False

    metric_cols = [c for c in IR_METRICS if c in df_submission.columns]
    if not metric_cols:
        return False

    nc = _name_col(df_submission)
    _, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_cols))
    w = 0.8 / max(len(df_submission), 1)

    for i, (_, row) in enumerate(df_submission.iterrows()):
        vals = [row[c] for c in metric_cols]
        ax.bar(x + i * w, vals, w, label=str(row[nc])[:30], color=PAL[i % len(PAL)], alpha=0.8)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * w, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + w * (len(df_submission) - 1) / 2)
    ax.set_xticklabels(metric_cols)
    ax.set_ylabel("Score")
    ax.set_title("Final Submission - JOKER Task 1 (Boosted Pipeline)")
    ax.legend(frameon=True, fontsize=8)
    save_plot("submission_results")
    return True


def print_summary(df_summary: pd.DataFrame) -> None:
    if not _available(df_summary):
        return

    rename_map = {
        "R@30": COL_RECALL30, "R@1000": "Recall@1000",
        "s1_recall": "stage1_recall", "s2_recall": "stage2_recall",
        "s3_recall": "stage3_recall",
    }
    df_summary = df_summary.rename(columns={k: v for k, v in rename_map.items() if k in df_summary.columns})

    print(f"\n{'=' * 70}")
    print("\t-> RESULTS SUMMARY")
    print(f"{'=' * 70}")

    display_cols = [
        c for c in [
            "run_name", "experiment", "config",
            COL_MAP, COL_NDCG10, COL_RECALL30,
            COL_P5, COL_P10, "stage1_recall",
            "stage2_recall", "stage3_recall"
        ]
        if c in df_summary.columns
    ]

    summary = df_summary[display_cols].copy()
    sort_col = COL_MAP if COL_MAP in summary.columns else summary.columns[-1]
    summary = summary.sort_values(sort_col, ascending=False)
    summary.to_csv(os.path.join(PathManager.RESULTS_DIR, "summary_table.csv"), index=False)
    print(summary.to_string(index=False))

    metric_cols = [c for c in IR_METRICS if c in summary.columns]
    nc = _name_col(summary)
    if not metric_cols:
        return

    print(f"\n{'=' * 70}")
    print("\t-> LaTeX Table:")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Pipeline Results - JOKER Task 1}")
    header = " & ".join(["Configuration"] + metric_cols) + r" \\"
    print(r"\begin{tabular}{l" + "c" * len(metric_cols) + "}")
    print(LATEX_HLINE)
    print(header)
    print(LATEX_HLINE)
    for _, row in summary.iterrows():
        name = str(row.get(nc, ""))[:25].replace("_", " ")
        vals = " & ".join(f"{row[c]:.4f}" for c in metric_cols)
        print(f"{name} & {vals} " + r"\\")
    print(LATEX_HLINE)
    print(r"\end{tabular}")
    print(r"\end{table}")


def main() -> None:
    """
    Generates plots from experiment CSVs:
      1. Round 1 vs Round 2 comparison
      2. Stage-level recall analysis
      3. Pipeline leaderboard (all runs)
      4. Cross-encoder comparison (Table 2)
      5. Cross-encoder training curves (loss per epoch)
      6. Dense embedder comparison (Table 3)
      7. Judge comparison (Table 4)
      8. Judge training curves
      9. Ablation study leaderboard
     10. Dataset statistics overview
     11. Submission summary
     12. Summary table (console + LaTeX)

    Usage:
        python evaluate_plots.py
    """
    ensure_dirs()
    print("Loading results...")

    df_r1 = load_csv("pipeline_experiments.csv")
    df_r2 = load_csv("pipeline_improved_experiments.csv")
    df_boosted = load_csv("boosted/pipeline_results.csv")
    df_submission = load_csv("eval/pipeline_results.csv")

    ce_experiment_dfs = [load_csv("ce_experiments.csv")]

    ce_hists = {n: load_csv(f"ce_{n}_history.csv") for n in [
        "CE_MiniLM_new", "CE_MiniLM_new_aug",
        "CE_BGE_new", "CE_BGE_new_aug",
        "CE_GTE_new", "CE_GTE_new_aug",
    ]}

    judge_experiment_dfs = [load_csv("judge_experiments.csv")]

    judge_curve_dfs = [
        load_csv(f"judge_{name}_curves.csv") for name in [
            "Judge_Qwen7B_new", "Judge_Qwen7B_new_aug",
            "Judge_Qwen7B_old", "Judge_Qwen7B_old_aug",
            "Judge_G4_31B_new", "Judge_G4_31B_new_aug",
            "Judge_G4_31B_old", "Judge_G4_31B_old_aug",
        ]
    ]

    df_embedder = load_csv("embedder_search/embedder_results.csv")

    df_stats = load_csv("data_statistics.csv")
    df_paper = load_csv("paper_summary_table.csv")
    df_ablation = load_csv("ablation/ablation_results.csv")

    results = [
        plot_round_comparison(df_r1, df_r2),
        plot_stage_recall(df_r1, df_r2),
        plot_pipeline_leaderboard(_first_non_empty(df_boosted, df_r2, df_r1)),
        plot_ce_comparison(ce_experiment_dfs),
        plot_ce_training_curves(ce_hists),
        plot_embedder_comparison(df_embedder),
        plot_judge_comparison(judge_experiment_dfs),
        plot_judge_curves(judge_curve_dfs),
        plot_ablation(df_ablation),
        plot_dataset_stats(df_stats),
        plot_submission(df_submission),
    ]
    plots_created = sum(results)

    print_summary(_first_non_empty(df_paper, df_boosted, df_r2, df_r1))

    if plots_created == 0:
        print("\n!\tNo result CSVs found. Run experiments first, then re-run this script.")
    else:
        print(f"\nDone: {plots_created} plots saved to {PathManager.PLOT_DIR}/")


if __name__ == "__main__":
    main()

import os
from pathlib import Path

from iroh.core.path_manager import PathManager

os.environ["PYTHONUTF8"] = "1"
import sys
import time
import json
import subprocess
import argparse
import datetime

STAGES = {
    1: "Data Processing",
    2: "Train Cross-Encoders",
    3: "Train Judges (Gemma 4)",
    4: "Score Corpus Humor",
    5: "Precompute Embeddings",
    6: "Local Evaluation",
    7: "Ablation Study",
    8: "Generate Plots",
    9: "CodaBench Top-N Submissions",
    10: "Ensemble Weight Ablation",
}

_RUNNABLE = Path(__file__).resolve().parent


def _script(name: str) -> str:
    return str(_RUNNABLE / name)


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg, icon="[i]"):
    print(f"[{ts()}] {icon} {msg}")


def banner(stage_num, name):
    print()
    print(f"{'-' * 70}")
    print(f"\tSTAGE {stage_num}: {name.upper()}")
    print(f"{'-' * 70}")


def run(cmd, description, dry_run=False):
    """Run a subprocess, stream output, return success bool."""
    cmd_str = " ".join(cmd)
    log(f"{description}", "[-]")
    print(f"\t$ {cmd_str}")

    if dry_run:
        log("(dry run - skipped)", "[-]")
        return True

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        log(f"{description} - done ({elapsed / 60:.1f} min)", "[OK]")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        log(f"{description} - FAILED (exit {e.returncode}) after {elapsed / 60:.1f} min", "[X]")
        return False
    except FileNotFoundError:
        log(f"Script not found: {cmd[1]}", "[X]")
        return False


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            log(f"GPU: {name} ({mem:.0f} GB)", "")
            return True
        else:
            log("No GPU detected", "[!]")
            return False
    except ImportError:
        return False


def check_data_files(base_dir):
    """Verify required data files exist."""
    data_dir = str(PathManager.DATA_DIR)
    required = [
        ("joker_task1_retrieval_corpus26_english.json", True),
        ("joker_task1_retrieval_queries_test26_english.json", True),
        ("joker_task1_retrieval_qrels_train26_english.json", True),
    ]
    optional = [
        "processed_joker_train.json",
        "temp_step1_rationales.json",
        "temp_step2_augmented.json",
        "old_temp_step1_rationales.json",
        "old_temp_step2_augmented.json",
    ]

    ok = True
    for fname, is_required in required:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            log(f"\t[OK] {fname} ({size_mb:.1f} MB)", "")
        elif is_required:
            log(f"\t[x] MISSING: {fname}", "[X]")
            ok = False

    for fname in optional:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    n = len(data)
                except Exception:
                    n = "?"
            log(f"\t[OK] {fname} ({n} items)", "")
        else:
            log(f"\t[!] {fname} (not found, will skip configs that need it)", "[!]")

    return ok


def stage_1(args, base_dir):
    """Data processing - corpus, balanced sampling, splits, analysis plots."""
    return run(
        [sys.executable, _script("data_processing.py"), "--plots"],
        "Data processing + analysis plots",
        args.dry_run,
    )


def stage_2(args, base_dir):
    """Train cross-encoders - 6 configs (3 architectures x ±aug on new data, Table 2 in paper)."""
    return run(
        [sys.executable, _script("train_cross_encoder.py")],
        "Train cross-encoders (MiniLM-L6, BGE-Reranker-Base, GTE-Reranker-ModernBERT-Base x ±aug)",
        args.dry_run,
    )


def stage_3(args, base_dir):
    """Train judges - Qwen2.5-7B + Gemma 4-31B QLoRA, 8 configs (Table 4 in paper)."""
    cmd = [sys.executable, _script("train_judge_gemma4.py")]
    if getattr(args, "gemma_only", False):
        cmd.append("--gemma")
        desc = "Train judges (Gemma 4 31B x 4 configs: new/new_aug/old/old_aug)"
    elif getattr(args, "qwen_only", False):
        cmd.append("--qwen")
        desc = "Train judges (Qwen 7B x 4 configs: new/new_aug/old/old_aug)"
    else:
        desc = "Train judges (Qwen 7B + Gemma 4 31B x 4 data variants = 8 configs)"
    return run(cmd, desc, args.dry_run)


def stage_4(args, base_dir):
    """Score entire corpus with humor prior."""
    return run(
        [sys.executable, _script("score_corpus_humor.py")],
        "Score corpus humor prior (full corpus inference)",
        args.dry_run,
    )


def stage_5(args, base_dir):
    """Precompute dense embeddings - cache on CPU before pipeline runs."""
    return run(
        [sys.executable, _script("precompute_embeddings.py")],
        "Precompute corpus embeddings (cached to CPU)",
        args.dry_run,
    )


def stage_6(args, base_dir):
    """Local evaluation - pipeline on held-out test split."""
    cmd = [sys.executable, _script("pipeline.py"), "--results-subdir", "eval"]
    if getattr(args, "no_judge", False):
        cmd.append("--no-judge")
    return run(
        cmd,
        "Local pipeline evaluation (held-out test split)",
        args.dry_run,
    )


def stage_7(args, base_dir):
    """Ablation study - all CE x Judge combinations."""
    return run(
        [sys.executable, _script("ablation.py")],
        "Ablation study (all CE x Judge combinations)",
        args.dry_run,
    )


def stage_8(args, base_dir):
    """Generate evaluation plots."""
    return run(
        [sys.executable, _script("evaluate_plots.py")],
        "Generate publication plots",
        args.dry_run,
    )


def stage_9(args, base_dir):
    """CodaBench top-N submissions - top configs from ablation -> full test queries."""
    top_n = getattr(args, "top_n", 30)
    cmd = [sys.executable, _script("pipeline.py"), "--top-n", str(top_n),
           "--results-subdir", "top_submissions"]
    if getattr(args, "no_judge", False):
        cmd.append("--no-judge")
    success = run(
        cmd,
        f"CodaBench top-{top_n} submissions (ablation winners -> all test queries)",
        args.dry_run,
    )
    if not success or args.dry_run:
        return success

    prediction_path = os.path.join(base_dir, "prediction.json")
    zip_path = os.path.join(base_dir, "prediction.zip")

    if os.path.exists(prediction_path):
        try:
            if not os.path.exists(zip_path):
                import zipfile
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(prediction_path, "prediction.json")
            with open(prediction_path, "r") as f:
                data = json.load(f)
            n_queries = len(set(item["qid"] for item in data))
            log(f"Best submission packaged: {zip_path}", "📦")
            log(f"\t{len(data)} records, {n_queries} queries", "[i]")
            log(f"\tUpload {zip_path} to CodaBench!", "[$]")
        except Exception as e:
            log(f"Failed to package submission: {e}", "[!]")
    else:
        log("prediction.json not found - check pipeline output", "[!]")

    return success


def stage_10(args, base_dir):
    """Ensemble weight ablation - sweep Qwen/Gemma weights and CE/Judge blends."""
    cmd = [sys.executable, _script("ensemble_ablation.py")]
    if getattr(args, "blend_only", False):
        cmd.append("--blend-only")
    if getattr(args, "no_save_ens", False):
        cmd.append("--no-save")
    if getattr(args, "ce_name_ens", None):
        cmd += ["--ce-name", args.ce_name_ens]
    if getattr(args, "stage1_ver", None):
        cmd += ["--stage1-ver", args.stage1_ver]
    if getattr(args, "ce_ver", None):
        cmd += ["--ce-ver", args.ce_ver]
    if getattr(args, "judge_ver", None):
        cmd += ["--judge-ver", args.judge_ver]
    return run(
        cmd,
        "Ensemble weight ablation (Qwen x Gemma-old x Gemma-new x CE/J blend)",
        args.dry_run,
    )


RUNNERS = {
    1: stage_1, 2: stage_2, 3: stage_3, 4: stage_4, 5: stage_5,
    6: stage_6, 7: stage_7, 8: stage_8, 9: stage_9, 10: stage_10,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="JOKER Pipeline - Full Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  Data Processing             6  Local Evaluation
  2  Train Cross-Encoders        7  Ablation Study (CE with Judge)
  3  Train Judges (Gemma 4)      8  Generate Plots
  4  Score Corpus Humor          9  CodaBench Top-N Submissions
  5  Precompute Embeddings       10 Ensemble Weight Ablation

CE experiment matrix (Table 2 in paper) - 3 architectures x ±aug on new data:
  Model                         CE name          Augmentation
  --------------------------    -------------    ------------
  MiniLM-L-6 (22M)             CE_MiniLM_new    False
  MiniLM-L-6 (22M)             CE_MiniLM_new_aug True
  BGE-Reranker-Base (278M)     CE_BGE_new       False
  BGE-Reranker-Base (278M)     CE_BGE_new_aug   True
  GTE-Reranker-ModernBERT-Base CE_GTE_new       False  <- paper best (MAP 0.2843)
  GTE-Reranker-ModernBERT-Base CE_GTE_new_aug   True

Judge experiment matrix (Table 4 in paper) - 2 models x 4 data variants:
  Model              Judge name              Data
  -----------------  ----------------------  -------------
  Qwen2.5-7B         Judge_Qwen7B_new        new
  Qwen2.5-7B         Judge_Qwen7B_new_aug    new + aug
  Qwen2.5-7B         Judge_Qwen7B_old        old          <- paper best (MAP 0.6055)
  Qwen2.5-7B         Judge_Qwen7B_old_aug    old + aug
  Gemma-4-31B        Judge_G4_31B_new        new
  Gemma-4-31B        Judge_G4_31B_new_aug    new + aug
  Gemma-4-31B        Judge_G4_31B_old        old
  Gemma-4-31B        Judge_G4_31B_old_aug    old + aug

Stage 9 flow:
  Ablation (stage 7) ranks all CE with Judge combos on the local test split.
  Stage 9 takes the top N (default 30) and generates a full CodaBench
  submission for each using ALL official test queries. The best one is
  packaged as prediction.zip for upload.

Examples:
  python run_all.py                     # full run (top 30 submissions)
  python run_all.py --top-n 10          # only top 10 submissions
  python run_all.py --from-stage 3      # resume from judge training
  python run_all.py --only 7 9          # ablation + top-30 submissions
  python run_all.py --skip 7            # skip ablation (can be slow)
  python run_all.py --dry-run           # show plan without executing
        """,
    )
    parser.add_argument("--from-stage", type=int, default=None, help="Start from this stage (inclusive)")
    parser.add_argument("--only", type=int, nargs="+", default=None, help="Run ONLY these stages")
    parser.add_argument("--skip", type=int, nargs="+", default=None, help="Skip these stages")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top ablation configs to submit (default: 30)")
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip Stage 3 judge in pipeline runs (stages 8 & 11). Stages 1+2 only. In stage 11, dedupes top-N by CE."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop if any stage fails (default: continue)")
    parser.add_argument(
        "--blend-only", action="store_true",
        help="Stage 10: only test the best-known weight point x blends (~3 combos)"
    )
    parser.add_argument(
        "--no-save-ens", action="store_true", help="Stage 10: score combos but do not write submission JSONs"
    )
    parser.add_argument("--ce-name-ens", default=None, help="Stage 10: CE model name to use (default: CE_GTE_new)")
    parser.add_argument("--stage1-ver", default=None, help="Stage 10: stage-1 cache version tag (default: v5)")
    parser.add_argument("--ce-ver", default=None, help="Stage 10: CE cache version tag (default: v5)")
    parser.add_argument("--judge-ver", default=None, help="Stage 10: judge-prob cache version tag (default: v6)")
    parser.add_argument(
        "--e4b", action="store_true",
        help="Log the run as using the lightweight Gemma 4 E4B judge variant (informational only)."
    )
    return parser.parse_args()


def main():
    """
    Runs everything except rationale/negative generation (Ollama).
    Assumes these files already exist in data/:
      - temp_step1_rationales.json, temp_step2_augmented.json         (new)
      - old_temp_step1_rationales.json, old_temp_step2_augmented.json (old)
      - processed_joker_train.json                                    (base)

    Stages:
      1. Data processing        - corpus, balanced sampling, train/test split
      2. Train cross-encoders   - 6 configs (3 architectures with/out aug, Table 2 in paper)
      3. Train judges           - 8 configs (Qwen7B + Gemma4-31B with 4 data variants, Table 4)
      4. Score corpus humor     - Judge inference on full corpus
      5. Precompute embeddings  - Dense retriever embeddings cached to CPU
      6. Local evaluation       - Pipeline on held-out test split (best CE + Judge)
      7. Ablation study         - All CE with Judge combinations (local test split)
      8. Generate plots         - Publication-quality figures
      9. Top-N submissions      - Top 30 ablation configs -> full test queries -> CodaBench
      10. Ensemble ablation     - Sweep Qwen/Gemma weight ratios + CE/J blends (CPU-only)

    Usage:
        python run_all.py                        # full run (top 30 submissions)
        python run_all.py --top-n 10             # only top 10 submissions
        python run_all.py --from-stage 3         # resume from judge training
        python run_all.py --only 7 9             # ablation + top-30 submissions
        python run_all.py --only 10              # ensemble weight ablation only
        python run_all.py --only 10 --blend-only # fast: best-known weights with 3 blends
        python run_all.py --skip 7               # skip ablation (slow)
        python run_all.py --dry-run              # print plan only
    """
    args = parse_args()
    base_dir = PathManager.BASE

    print()
    print("=" * 70)
    print("\tJOKER Pipeline - Full Runner")
    print("\tGemma 4 + All Data Variants (new / old / combined / base)")
    print("=" * 70)
    log(f"Base: {base_dir}")
    check_gpu()
    if args.e4b:
        log("Judge: Gemma 4 E4B (lightweight, 2 configs)")
    else:
        log("Judge: Gemma 4 31B (7 configs: new/old/combined/base x ±aug)")
    print()

    log("Checking data files...", "[*]")
    if not check_data_files(base_dir):
        log("Missing required data files - cannot proceed", "[X]")
        return
    print()

    all_stages = sorted(STAGES.keys())

    if args.only is not None:
        to_run = sorted(set(args.only))
    elif args.from_stage is not None:
        to_run = [s for s in all_stages if s >= args.from_stage]
    else:
        to_run = list(all_stages)

    if args.skip:
        to_run = [s for s in to_run if s not in args.skip]

    print("-" * 70)
    print("\tEXECUTION PLAN")
    print("-" * 70)
    for s in all_stages:
        marker = " [Y]" if s in to_run else " [N]"
        print(f"\t{marker} Stage {s}: {STAGES[s]}")
    print("-" * 70)
    print()

    if not to_run:
        log("No stages to run!", "[!]")
        return

    results = {}
    total_start = time.time()

    for stage_num in to_run:
        banner(stage_num, STAGES[stage_num])

        stage_start = time.time()
        success = RUNNERS[stage_num](args, base_dir)
        stage_elapsed = time.time() - stage_start

        results[stage_num] = {
            "name": STAGES[stage_num],
            "success": success,
            "time_sec": round(stage_elapsed, 1),
        }

        if not success and args.stop_on_fail:
            log(f"Stopping - stage {stage_num} failed (--stop-on-fail)", "[X]")
            break

    total_elapsed = time.time() - total_start

    print()
    print("=" * 70)
    print("\tPIPELINE SUMMARY")
    print("=" * 70)
    print()

    for stage_num, r in results.items():
        icon = "[OK]" if r["success"] else "[X]"
        t = r["time_sec"]
        time_str = f"{t / 60:.1f} min" if t > 60 else f"{t:.0f}s"
        print(f"\t{icon} Stage {stage_num}: {r['name']:<35} {time_str:>10}")

    total_hr = total_elapsed / 3600
    total_min = total_elapsed / 60
    print()
    if total_hr >= 1:
        print(f"\tTotal: {total_hr:.1f} hr ({total_min:.0f} min)")
    else:
        print(f"\tTotal: {total_min:.1f} min")

    failed = [s for s, r in results.items() if not r["success"]]
    if failed and not args.dry_run:
        print(f"\n\tFailed: stages {failed}")
        print(f"\tResume: python run_all.py --from-stage {failed[0]}")
    elif not args.dry_run:
        print("\n\t[OK] All stages completed successfully!")

        zip_path = os.path.join(base_dir, "prediction.zip")
        if os.path.exists(zip_path):
            size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            print(f"\n\tSubmission: {zip_path} ({size_mb:.1f} MB)")
            print(f"\t\t\t -> Upload to CodaBench")

        results_dir = os.path.join(base_dir, "results")
        for csv_name in [
            "ce_experiments.csv",
            "judge_gemma4_experiments.csv",
            "ablation/ablation_results.csv",
            "ensemble_ablation/ensemble_ablation_results.csv",
            "top_submissions/submission_tracker.csv"
        ]:
            csv_path = os.path.join(results_dir, csv_name)
            if os.path.exists(csv_path):
                print(f"\t > {csv_name}")

    print()


if __name__ == "__main__":
    main()

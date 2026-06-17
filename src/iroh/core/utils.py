import gc
import json
import math
import os
import random
import re
from typing import LiteralString

import matplotlib
import numpy as np
import torch
from nltk.stem import PorterStemmer

from iroh.core.config import SEED, STOP_WORDS
from iroh.core.path_manager import PathManager

_stemmer = PorterStemmer()
EARLY_STOP_SENTINEL = "early_stopped.flag"


def seed_everything(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_stemmed(text: str) -> list[str]:
    """Tokenize and stem text for BM25 indexing."""
    if not text:
        return []
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    return [_stemmer.stem(t) for t in tokens]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


def expand_query(query_text: str) -> list[str]:
    """
    Generate query expansions for humor-oriented retrieval.

    Always applies enhanced expansions (double meaning, Tom Swifty detection).
    """
    expansions = [query_text, f"{query_text} joke pun wordplay", f"{query_text} funny humor"]

    words = query_text.lower().split()
    content = [w for w in words if w not in STOP_WORDS]
    if content:
        expansions.append(" ".join(content))
        expansions.append(f"{' '.join(content)} pun double meaning")

    if any(w in query_text.lower() for w in ["said", "tom", "adverb"]):
        expansions.append(f"said Tom {query_text} swifty adverb")

    return expansions[:6]


def load_json(path: str) -> list | dict:
    """Load a JSON file. Supports both JSON arrays and JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def save_json(data, path: str):
    """Save data as a formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_results_csv(csv_path: str, record: dict):
    """
    Append a result record to a CSV, creating it if it doesn't exist.
    """
    import pandas as pd
    existing = []
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path).to_dict("records")
    existing.append(record)
    pd.DataFrame(existing).to_csv(csv_path, index=False)


def load_corpus(corpus_file: str) -> tuple[list[dict], list[str], list[str]]:
    """
    Load the JOKER corpus JSON and return the raw list plus parallel text and docid arrays.

    :param corpus_file: Path to which corpus to use
    :return: (corpus, corpus_texts, corpus_docids)
    """
    corpus = load_json(corpus_file)
    corpus_texts = [doc.get("text", "") for doc in corpus]
    corpus_docids = [str(doc.get("docid", f"doc_{i}")) for i, doc in enumerate(corpus)]
    return corpus, corpus_texts, corpus_docids


def build_qrel_dict(qrels_data: list[dict]) -> dict[str, set[str]]:
    """
    Convert a flat list of qrel records into a dict mapping qid to set of relevant docids.
    """
    qrel_dict: dict[str, set[str]] = {}
    for item in qrels_data:
        qrel_dict.setdefault(str(item["qid"]), set()).add(str(item["docid"]))
    return qrel_dict


def load_humor_prior(corpus_docids: list[str]) -> tuple[dict | list, np.ndarray]:
    """
    Load cached humor prior scores and return both the raw dict and
    a numpy array aligned to corpus_docids (defaulting to 0.5).
    """
    humor_prior: dict | list = {}
    if os.path.exists(PathManager.CORPUS_HUMOR_CACHE):
        humor_prior = load_json(PathManager.CORPUS_HUMOR_CACHE)
    humor_prior_array = np.array([humor_prior.get(d, 0.5) for d in corpus_docids])
    return humor_prior, humor_prior_array


def safe_mean(values: list) -> float:
    """Return the rounded mean of a list, or 0.0 if the list is empty."""
    return round(float(np.mean(values)), 4) if values else 0.0


def stage_recall_record(stage_recalls: dict) -> dict[str, float]:
    """
    Convert a {1: [...], 2: [...], 3: [...]} stage-recall dict into the flat keys used in result records.
    """
    return {
        "s1_recall": safe_mean(stage_recalls[1]),
        "s2_recall": safe_mean(stage_recalls[2]),
        "s3_recall": safe_mean(stage_recalls[3]),
    }


def merge_extra_data(base_data: list, extra_file_path: str | None, log_label: str = "") -> list:
    if extra_file_path and os.path.exists(extra_file_path):
        extra = load_json(extra_file_path)
        existing_texts = {item.get("text", "") for item in base_data}
        new_items = [item for item in extra if item.get("text", "") not in existing_texts]
        base_data.extend(new_items)
        label_str = f"{log_label} " if log_label else ""
        print(f"\t\t+ Merged {len(new_items)} {label_str}from {os.path.basename(extra_file_path)} "
              f"(total: {len(base_data)})")
    return base_data


def load_data_for_config(config: dict, fallback_rationale: str, fallback_augmented: str):
    """
    Load original + augmented data based on a training config's file overrides.

    Configs can specify:
      - rationale_file: path to the rationale/original data file
      - augmented_file: path to augmented data file (or None)
      - extra_rationale_file: additional rationale file to merge (for "combined" runs)
      - extra_augmented_file: additional augmented file to merge

    :return: (original_data: list, aug_data: list)
    """
    rat_path = config.get("rationale_file", fallback_rationale)
    if not os.path.exists(rat_path):
        rat_path = fallback_rationale
    original_data = load_json(rat_path) if os.path.exists(rat_path) else []
    print(f"\t\tOriginal data: {len(original_data)} from {os.path.basename(rat_path)}")

    extra_rat = config.get("extra_rationale_file")
    merge_extra_data(original_data, extra_rat)

    aug_data = []
    aug_path = config.get("augmented_file", fallback_augmented)
    if aug_path and os.path.exists(aug_path):
        aug_data = load_json(aug_path)
        print(f"\t\tAugmented data: {len(aug_data)} from {os.path.basename(aug_path)}")

    extra_aug = config.get("extra_augmented_file")
    merge_extra_data(aug_data, extra_aug, log_label="aug")

    return original_data, aug_data


def compute_yes_probability(
        model,
        tokenizer,
        yes_id: int,
        no_id: int,
        query: str,
        text: str,
        system_prompt: str = "You are a humor and wordplay detection judge. Answer only YES or NO.",
        max_length: int = 512,
) -> float:
    """Run the LoRA judge model on a (query, text) pair. Returns P(YES)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f'Query: "{query}"\n'
            f'Text: "{text}"\n'
            "Is this a relevant joke? Answer YES or NO."
        )},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]

    y_l = logits[0, yes_id].item()
    n_l = logits[0, no_id].item()
    mx_l = max(y_l, n_l)
    return math.exp(y_l - mx_l) / (math.exp(y_l - mx_l) + math.exp(n_l - mx_l))


def get_yes_no_ids(tokenizer):
    """Get token IDs for YES and NO from a tokenizer."""
    yes_id = tokenizer.encode(" YES", add_special_tokens=False)[-1]
    no_id = tokenizer.encode(" NO", add_special_tokens=False)[-1]
    return yes_id, no_id


def find_best_model(models_dir: str, candidates: list[str]) -> str | None:
    """Return the path of the first available model from a priority-ordered list."""
    for name in candidates:
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_judge_model(judge_path: str, base_model_id: str | None = None):
    """
    Load a quantised judge model with LoRA adapter. Compatible with both Qwen and Gemma 4.

    Auto-detects the base model from judge_meta.json saved during training.
    Falls back to base_model_id if no metadata found.

    JUDGE_BASE_MODEL in config.py is a last-resort fallback only.
    The authoritative base model is stored in judge_meta.json at training time.

    :return: (model, tokenizer, yes_id, no_id)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    meta_path = os.path.join(judge_path, "judge_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        resolved_id = meta.get("base_model_id", base_model_id)
        if resolved_id != base_model_id:
            print(f"\tjudge_meta.json: using {resolved_id} (not {base_model_id})")
        base_model_id = resolved_id

    if base_model_id is None:
        raise ValueError(
            f"No base_model_id provided and no judge_meta.json found in {judge_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    extra_kwargs = {}
    if "gemma" in base_model_id.lower():
        extra_kwargs["attn_implementation"] = "eager"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        **extra_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, judge_path)
    model.eval()

    yes_id, no_id = get_yes_no_ids(tokenizer)
    return model, tokenizer, yes_id, no_id


def free_gpu(*objects):
    """Delete objects and clear GPU memory."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_plot_style():
    """
    Apply the shared publication matplotlib style used by data_processing.py and evaluate_plots.py.
    Call once at the top of any plotting script.
    """
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def evaluate_trec(
        submission_data: list[dict],
        qrels_data: list[dict],
        metrics: set[str] | None = None,
) -> dict[str, float]:
    """Evaluate a submission against qrels using pytrec_eval."""
    if metrics is None:
        from iroh.core.config import EVAL_METRICS
        metrics = EVAL_METRICS

    try:
        import pytrec_eval
    except ImportError:
        print("WARNING: pytrec_eval not installed, returning zeros.")
        return {m: 0.0 for m in metrics}

    qrel_eval = {}
    for item in qrels_data:
        qrel_eval.setdefault(str(item["qid"]), {})[str(item["docid"])] = int(item["qrel"])

    run_eval = {}
    for item in submission_data:
        run_eval.setdefault(str(item["qid"]), {})[str(item["docid"])] = float(item["score"])

    evaluator = pytrec_eval.RelevanceEvaluator(qrel_eval, metrics)
    results = evaluator.evaluate(run_eval)

    aggregated = {}
    for metric in metrics:
        vals = [q[metric] for q in results.values()]
        aggregated[metric] = float(np.mean(vals)) if vals else 0.0
    return aggregated


def mark_early_stopped(save_dir: str):
    """
    Write a sentinel file so future runs know this model finished via early stopping and must not be re-trained.
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, EARLY_STOP_SENTINEL), "w") as f:
        f.write("early_stopped\n")


def check_training_status(save_dir) -> tuple[str, str | None | LiteralString | bytes]:
    """
    Analyzes a directory to see if we should skip, resume, or start fresh.
    Status can be: 'skip', 'resume', 'start'

    Skip conditions (model is considered complete, do not retrain):
      - early_stopped.flag is present then training was stopped by patience
      - config + weights both exist then training ran to completion

    Resume condition:
      - HuggingFace checkpoint-N sub-directories exist but no complete model yet.

    :return: (status, checkpoint_path)
    """
    if not os.path.exists(save_dir):
        return 'start', None

    if os.path.exists(os.path.join(save_dir, EARLY_STOP_SENTINEL)):
        return 'skip', None

    has_config = os.path.exists(os.path.join(save_dir, "config.json")) or \
                 os.path.exists(os.path.join(save_dir, "adapter_config.json"))
    has_weights = os.path.exists(os.path.join(save_dir, "model.safetensors")) or \
                  os.path.exists(os.path.join(save_dir, "adapter_model.safetensors"))

    if has_config and has_weights:
        return 'skip', None

    checkpoints = [d for d in os.listdir(save_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        return 'resume', os.path.join(save_dir, latest_checkpoint)

    return 'start', None

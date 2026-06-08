import argparse
import gc
import os
import random
import time

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

from config import (
    SEED, JUDGE_BASE_MODEL, JUDGE_LORA_R, JUDGE_LORA_ALPHA,
    JUDGE_MAX_LENGTH, JUDGE_LORA_TARGETS,
    JUDGE_SYSTEM_PROMPT, JUDGE_TRAIN_CONFIGS,
    JUDGE_GEMMA_31B_CONFIGS, JUDGE_QWEN7B_CONFIGS,
    ensure_dirs,
)
from path_manager import PathManager
from utils import seed_everything, get_yes_no_ids, load_data_for_config, check_training_status, mark_early_stopped


class MetricsRecorder(TrainerCallback):
    def __init__(self):
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            self.records.append({
                "step": state.global_step,
                "epoch": logs.get("epoch", 0),
                "eval_loss": logs["eval_loss"],
            })


def make_format_fn(tokenizer):
    def format_chat(example):
        target = "YES" if example["label"] == 1 else "NO"
        query = example.get("query", "General Wordplay")
        text = example.get("text", "")
        user_msg = (
            f"Evaluate if the following text is BOTH relevant to the user query "
            f"AND contains humor/wordplay.\n\n"
            f'Query: "{query}"\nText: "{text}"\n\n'
            f"Is this a relevant joke? Answer strictly YES or NO."
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": target},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": formatted}

    return format_chat


def test_inference_accuracy(model, tokenizer, val_data, n_test: int = 100) -> dict:
    yes_id, no_id = get_yes_no_ids(tokenizer)
    correct = yes_preds = no_preds = 0
    n_test = min(n_test, len(val_data))
    model.eval()

    for i in range(n_test):
        item = val_data[i]
        msgs = [
            {"role": "system", "content": "You are a humor and wordplay detection judge. Answer only YES or NO."},
            {"role": "user", "content": (
                f"Query: \"{item.get('query', 'General Wordplay')}\"\n"
                f"Text: \"{item.get('text', '')}\"\n"
                f"Is this a relevant joke?"
            )},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        predicted = 1 if logits[0, yes_id].item() > logits[0, no_id].item() else 0
        yes_preds += predicted
        no_preds += 1 - predicted
        if predicted == item["label"]:
            correct += 1

    return {
        "accuracy": correct / n_test, "correct": correct, "total": n_test,
        "yes_preds": yes_preds, "no_preds": no_preds,
    }


def load_gemma4_for_training(model_id, lora_r, lora_alpha, lora_targets):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except AttributeError as e:
        if "keys" in str(e):
            print(f"\tTokenizer workaround: patching extra_special_tokens...")
            from transformers import GemmaTokenizerFast
            import json as _json
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        else:
            raise
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    try:
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=lora_targets,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    except (ValueError, KeyError) as e:
        print(f"\tNamed LoRA targets failed ({e}), falling back to 'all-linear'")
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules="all-linear",
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\tTrainable: {trainable:,} / {total_p:,} ({100 * trainable / total_p:.2f}%)")
    return model, tokenizer


def train_one_config(config: dict) -> dict:
    """Train a single judge configuration with per-config data files."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {config['name']}")
    print(f"\tMax epochs: {config['max_epochs']}, Patience: {config['patience']}")
    print(f"{'=' * 60}")

    start_time = time.time()

    model_id = config.get("model_id", JUDGE_BASE_MODEL)
    lora_r = config.get("lora_r", JUDGE_LORA_R)
    lora_alpha = config.get("lora_alpha", JUDGE_LORA_ALPHA)
    max_length = config.get("max_length", JUDGE_MAX_LENGTH)
    lora_targets = config.get("lora_targets", JUDGE_LORA_TARGETS)

    print(f"\tModel: {model_id}, LoRA r={lora_r}")

    original_data, aug_data = load_data_for_config(
        config,
        fallback_rationale=PathManager.RATIONALES_FILE,
        fallback_augmented=PathManager.AUGMENTED_FILE,
    )

    if not original_data:
        print(f"\tSKIPPING - no training data found")
        return {"experiment": config["name"], "error": "no data"}

    model, tokenizer = load_gemma4_for_training(model_id, lora_r, lora_alpha, lora_targets)
    format_chat = make_format_fn(tokenizer)

    original_ds = Dataset.from_list(original_data)
    split = original_ds.train_test_split(test_size=0.1, seed=SEED)
    train_original_raw = split["train"]
    val_dataset_raw = split["test"]

    print(f"\tVal set (clean): {len(val_dataset_raw)}")

    if config["use_augmented"] and aug_data:
        combined = list(train_original_raw) + aug_data
        positives = [x for x in combined if x.get("label") == 1]
        negatives = [x for x in combined if x.get("label") == 0]
        if len(positives) < len(negatives):
            factor = len(negatives) // len(positives)
            combined = positives * factor + negatives
        random.shuffle(combined)
        train_ds_raw = Dataset.from_list(combined)
        print(f"\tTraining with augmented: {len(train_ds_raw)}")
    else:
        train_ds_raw = train_original_raw

    train_dataset = train_ds_raw.map(format_chat)
    val_dataset = val_dataset_raw.map(format_chat)
    print(f"\tTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

    checkpoint_dir = os.path.join(PathManager.MODELS_DIR, f"{config['name']}_ckpt")
    save_dir = os.path.join(PathManager.MODELS_DIR, config["name"])
    os.makedirs(save_dir, exist_ok=True)

    recorder = MetricsRecorder()

    sft_config = SFTConfig(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config["grad_accum"],
        optim="paged_adamw_32bit",
        learning_rate=config["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        num_train_epochs=config["max_epochs"],
        fp16=False, bf16=True,
        max_length=max_length,
        dataset_text_field="text",
        seed=SEED,
        save_total_limit=3,
    )

    trainer = SFTTrainer(
        model=model, train_dataset=train_dataset, eval_dataset=val_dataset,
        args=sft_config, processing_class=tokenizer,
        callbacks=[recorder, EarlyStoppingCallback(early_stopping_patience=config["patience"])],
    )

    result = trainer.train()
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    actual_epochs = len(recorder.records)
    if actual_epochs < config["max_epochs"]:
        print(f"\tEarly stopped after {actual_epochs} epochs - writing sentinel.")
        mark_early_stopped(save_dir)

    import json
    meta = {
        "base_model_id": model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "config_name": config["name"],
    }
    with open(os.path.join(save_dir, "judge_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start_time

    train_loss = result.training_loss
    eval_loss = trainer.evaluate().get("eval_loss", float("inf"))

    best_epoch = 0
    if recorder.records:
        best_record = min(recorder.records, key=lambda r: r["eval_loss"])
        best_epoch = best_record["epoch"]

    print("\tTesting YES/NO accuracy...")
    inf_results = test_inference_accuracy(model, tokenizer, val_dataset_raw)

    metrics = {
        "experiment": config["name"],
        "model": model_id,
        "lora_r": lora_r,
        "data_variant": os.path.basename(config.get("rationale_file", "default")),
        "use_augmented": config["use_augmented"],
        "best_epoch": round(best_epoch, 1),
        "total_epochs": len(recorder.records),
        "train_loss": round(train_loss, 4),
        "eval_loss": round(eval_loss, 4),
        "inference_accuracy": round(inf_results["accuracy"], 4),
        "yes_preds": inf_results["yes_preds"],
        "no_preds": inf_results["no_preds"],
        "time_sec": round(elapsed, 1),
    }

    print(f"\n  {config['name']}:")
    print(f"\t\tBest epoch: {best_epoch:.0f}, Train: {train_loss:.4f}, Eval: {eval_loss:.4f}")
    print(f"\t\tAccuracy: {inf_results['accuracy']:.4f} "
          f"({inf_results['correct']}/{inf_results['total']})")

    if inf_results["yes_preds"] == 0:
        print("\t\tWARNING: predicts ALL as NO!")
    elif inf_results["no_preds"] == 0:
        print("\t\tWARNING: predicts ALL as YES!")

    curve_data = [{"experiment": config["name"], **r} for r in recorder.records]
    pd.DataFrame(curve_data).to_csv(
        os.path.join(PathManager.RESULTS_DIR, f"judge_{config['name']}_curves.csv"), index=False
    )

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main():
    """
    Fine-tune humor/relevance judges using QLoRA.
    Trains 8 configurations matching Table 4 in the paper: 2 models x 4 data variants (new, new+aug, old, old+aug).

    Usage:
        python train_judge_gemma4.py                     # train all 8 configs
        python train_judge_gemma4.py --config 0          # train only first config
        python train_judge_gemma4.py --gemma             # train only Gemma 4 31B (4 configs)
        python train_judge_gemma4.py --qwen              # train only Qwen 7B (4 configs)
    """
    parser = argparse.ArgumentParser(description="Train JOKER judge models")
    parser.add_argument(
        "--config", type=int, default=None, help="Train only config at this index (0-based)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gemma", action="store_true", help="Train only Gemma 4 31B judges (4 configs)")
    group.add_argument("--qwen", action="store_true", help="Train only Qwen 7B judges (4 configs)")
    args = parser.parse_args()

    seed_everything()
    ensure_dirs()

    if args.gemma:
        configs = JUDGE_GEMMA_31B_CONFIGS
        label = "Gemma 4 31B (4 configs: new/new_aug/old/old_aug)"
    elif args.qwen:
        configs = JUDGE_QWEN7B_CONFIGS
        label = "Qwen 2.5 7B (4 configs: new/new_aug/old/old_aug)"
    else:
        configs = JUDGE_TRAIN_CONFIGS
        label = "All judges - Qwen 2.5 7B + Gemma 4 31B (8 configs)"

    print(f"\n{label}")

    if args.config is not None:
        configs = [configs[args.config]]

    all_metrics = []
    for config in configs:
        save_dir = os.path.join(PathManager.MODELS_DIR, config["name"])
        status, _ = check_training_status(save_dir)
        if status == 'skip':
            print(f"\tSkipping '{config['name']}' - already completed or early stopped.")
            continue
        result = train_one_config(config)
        all_metrics.append(result)

    pd.DataFrame(all_metrics).to_csv(os.path.join(PathManager.RESULTS_DIR, "judge_experiments.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("Judge Training Complete")
    print(f"{'=' * 60}")
    for m in all_metrics:
        if "error" in m:
            print(f"\t{m['experiment']:<40} SKIPPED ({m['error']})")
        else:
            print(f"\t{m['experiment']:<40} best_ep={m['best_epoch']}  "
                  f"eval={m['eval_loss']:.4f}  acc={m['inference_accuracy']:.4f}")


if __name__ == "__main__":
    main()

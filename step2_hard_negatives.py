import argparse
import copy
import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import ollama

from path_manager import PathManager


def parse_args():
    parser = argparse.ArgumentParser(description="Generate hard negatives with Ollama")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model name")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Ollama workers")
    parser.add_argument("--input", default=os.path.join(PathManager.DATA_DIR, "temp_step1_rationales.json"))
    parser.add_argument("--output", default=os.path.join(PathManager.DATA_DIR, "temp_step2_augmented.json"))
    parser.add_argument("--skip-defused", action="store_true")
    parser.add_argument("--skip-wrong-topic", action="store_true")
    parser.add_argument("--skip-near-miss", action="store_true")
    return parser.parse_args()


def text_similarity(a, b):
    """Quick similarity check to reject bad generations."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def ollama_generate(prompt, model):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_gpu": 99},
        )
        text = response["message"]["content"].strip().replace('"', "").replace("'", "'")
        for prefix in ["Here is", "Sure,", "Here's", "Certainly"]:
            if text.lower().startswith(prefix.lower()):
                text = (
                    text.split("\n", 1)[-1].strip()
                    if "\n" in text
                    else text.split(":", 1)[-1].strip()
                )
        return text
    except Exception as e:
        print(f"\n[!] Ollama Error: {e}")
        return None


def generate_literal_rewrite(item, model):
    """Type 1: Same topic, no humor."""
    prompt = (
        f'Rewrite this joke/wordplay as a completely literal, factual, non-humorous statement.\n'
        f'Keep the same core topic and key subject words. Remove ALL humor, puns, and wordplay.\n'
        f'Output ONLY the rewritten text (one sentence), nothing else.\n\n'
        f'Original: "{item["text"]}"\nLiteral version:'
    )
    text = ollama_generate(prompt, model)
    if not text or len(text) < 10:
        return None
    if text_similarity(text, item["text"]) > 0.85:
        return None

    neg = copy.deepcopy(item)
    neg["text"] = text
    neg["label"] = 0
    neg["augmented"] = True
    neg["neg_type"] = "literal_rewrite"
    neg["rationale"] = "Literal rewrite - same topic, no humor."
    return neg


def generate_defused_joke(item, model):
    """Type 2: Joke structure kept, punchline ruined."""
    prompt = (
        f'Take this joke and slightly change it so the punchline no longer works.\n'
        f'Keep the setup and structure, but replace the KEY word/phrase that creates '
        f'the humor with a literal alternative.\n'
        f'The result should look like it COULD be a joke but isn\'t actually funny.\n'
        f'Output ONLY the modified text, nothing else.\n\n'
        f'Original joke: "{item["text"]}"\nDefused version:'
    )
    text = ollama_generate(prompt, model)
    if not text or len(text) < 10:
        return None
    if text_similarity(text, item["text"]) > 0.95:
        return None

    neg = copy.deepcopy(item)
    neg["text"] = text
    neg["label"] = 0
    neg["augmented"] = True
    neg["neg_type"] = "defused_joke"
    neg["rationale"] = "Joke with punchline neutralized."
    return neg


def generate_wrong_topic_joke(item, model):
    """Type 3: A real joke but on a completely different topic."""
    query = item.get("query", "the original topic")
    prompt = (
        f'Write a short, original one-liner joke or pun about a COMPLETELY DIFFERENT topic than "{query}".\n'
        f'The joke should be genuinely funny but have NOTHING to do with the original query.\n'
        f'Output ONLY the joke (one sentence), nothing else.\n\n'
        f'Original topic: "{query}"\nUnrelated joke:'
    )
    text = ollama_generate(prompt, model)
    if not text or len(text) < 10:
        return None

    neg = copy.deepcopy(item)
    neg["text"] = text
    neg["label"] = 0
    neg["augmented"] = True
    neg["neg_type"] = "wrong_topic_joke"
    neg["rationale"] = "Genuine joke but unrelated to the query."
    return neg


def generate_near_miss_pun(item, model):
    """Type 4 (NEW): Almost a pun but the wordplay doesn't quite work."""
    query = item.get("query", "the topic")
    prompt = (
        f'Create a sentence that TRIES to be a pun related to "{query}" but FAILS.\n'
        f'The sentence should use a word that sounds SIMILAR to a pun-worthy word but isn\'t actually a pun.\n'
        f'It should feel like a bad/forced attempt at humor that doesn\'t land.\n'
        f'Output ONLY the sentence, nothing else.\n\n'
        f'Topic: "{query}"\n'
        f'Example of a GOOD pun on this topic: "{item["text"]}"\n'
        f'Failed pun attempt:'
    )
    text = ollama_generate(prompt, model)
    if not text or len(text) < 10:
        return None

    neg = copy.deepcopy(item)
    neg["text"] = text
    neg["label"] = 0
    neg["augmented"] = True
    neg["neg_type"] = "near_miss_pun"
    neg["rationale"] = "Near-miss: attempts wordplay but the pun mechanism fails."
    return neg


def main():
    """
    Generates 4 types of hard negatives from each positive example:
      1. Literal rewrite: same topic, no humor
      2. Defused joke: joke structure kept, punchline ruined
      3. Wrong-topic joke: a real joke on a completely different topic
      4. Near-miss pun: almost a pun but the wordplay doesn't work

    Uses Gemma (or any Ollama model) with parallel workers. Saves progress every 25 tasks.

    Usage:
        python step2_hard_negatives.py [--model gemma4:e4b] [--workers 4]
        python step2_hard_negatives.py --skip-defused --skip-near-miss
    """
    args = parse_args()

    print(f"Step 2: Generating Hard Negatives with {args.model} ({args.workers} workers)\n")
    start_time = time.time()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    positives = [item for item in data if item.get("label") == 1]
    print(f"\tTotal items: {len(data)}")
    print(f"\tPositives (will generate negatives from): {len(positives)}")

    existing_texts = set()
    all_results = list(data)

    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for item in existing:
                if item.get("augmented"):
                    existing_texts.add(item.get("text", ""))
                    all_results.append(item)
            print(f"\tAlready generated: {len(existing_texts)} (will skip)")
        except Exception:
            pass

    generators = [("literal_rewrite", generate_literal_rewrite)]
    if not args.skip_defused:
        generators.append(("defused_joke", generate_defused_joke))
    if not args.skip_wrong_topic:
        generators.append(("wrong_topic_joke", generate_wrong_topic_joke))
    if not args.skip_near_miss:
        generators.append(("near_miss_pun", generate_near_miss_pun))

    tasks = []
    for item in positives:
        for neg_type, gen_func in generators:
            tasks.append((neg_type, item, gen_func))

    print(f"\tTasks to process: {len(tasks)}")

    completed = 0
    generated = 0
    duplicates = 0
    errors = 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    def process_task(task_tuple):
        neg_type, item, gen_func = task_tuple
        return neg_type, gen_func(item, args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_task, t): t for t in tasks}

        for future in as_completed(futures):
            completed += 1
            try:
                neg_type, result = future.result()
                if result is not None:
                    result_text = result.get("text", "")
                    if result_text not in existing_texts:
                        all_results.append(result)
                        existing_texts.add(result_text)
                        generated += 1
                    else:
                        duplicates += 1
            except Exception:
                errors += 1

            if completed % 25 == 0 or completed == len(tasks):
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)

                elapsed = time.time() - start_time
                avg = elapsed / completed if completed else 1
                eta = avg * (len(tasks) - completed)
                print(
                    f"\t[{completed}/{len(tasks)}] "
                    f"Generated: {generated} | Dupes: {duplicates} | "
                    f"Errors: {errors} | "
                    f"ETA: {datetime.timedelta(seconds=int(eta))}"
                )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    type_counts = {}
    for item in all_results:
        if item.get("augmented"):
            t = item.get("neg_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nStep 2 Complete!")
    print(f"\tOriginal items:     {len(data)}")
    print(f"\tNew negatives:      {generated}")
    print(f"\tDuplicates skipped: {duplicates}")
    for neg_type, count in sorted(type_counts.items()):
        print(f"\t  - {neg_type}: {count}")
    print(f"\tTotal dataset size: {len(all_results)}")
    print(f"\tTime: {datetime.timedelta(seconds=int(time.time() - start_time))}")
    print(f"\n\t\tCopy to Drive: {args.output} -> /content/drive/MyDrive/joker/data/")


if __name__ == "__main__":
    main()

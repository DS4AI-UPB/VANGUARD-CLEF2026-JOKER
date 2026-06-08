import argparse
import copy
import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama

from path_manager import PathManager

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gemma4:e4b', help='Ollama model name')
parser.add_argument('--workers', type=int, default=4, help='Parallel Ollama workers')
parser.add_argument('--input', default=os.path.join(PathManager.DATA_DIR, "temp_step1_rationales.json"))
parser.add_argument('--output', default=os.path.join(PathManager.DATA_DIR, "temp_step2_augmented.json"))
parser.add_argument('--skip-defused', action='store_true', help='Skip defused joke generation')
parser.add_argument('--skip-wrong-topic', action='store_true', help='Skip wrong-topic joke generation')
args = parser.parse_args()

os.makedirs('data', exist_ok=True)


def ollama_generate(prompt, model):
    """Single Ollama call with error handling."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'num_gpu': 99}
        )
        return response['message']['content'].strip().replace('"', '')
    except Exception as e:
        print(f"\n[!] Ollama Error: {e}")
        return None


def generate_literal_rewrite(item, model):
    """Type 1: Same topic, no humor. Teaches the model that topical relevance != humor."""
    prompt = f"""Rewrite the following joke/wordplay into a completely literal, factual, non-humorous statement.
Keep the same core topic and key subject words, but remove ALL humor, puns, and wordplay.
Output ONLY the rewritten text, nothing else.

Original: "{item['text']}"
Literal version:"""

    text = ollama_generate(prompt, model)
    if not text or len(text) < 10:
        return None

    neg = copy.deepcopy(item)
    neg['text'] = text
    neg['label'] = 0
    neg['augmented'] = True
    neg['neg_type'] = 'literal_rewrite'
    neg['rationale'] = "Literal rewrite of a joke - same topic, no humor."
    return neg


def generate_defused_joke(item, model):
    """Type 2: Joke structure kept, punchline ruined. Teaches subtle humor detection."""
    prompt = f"""Take this joke and slightly change it so the punchline no longer works. 
Keep the setup and structure, but replace the key word/phrase that makes it funny with something literal.
Output ONLY the modified text, nothing else.

Original joke: "{item['text']}"
Defused version:"""

    text = ollama_generate(prompt, model)
    if not text or len(text) < 10:
        return None

    neg = copy.deepcopy(item)
    neg['text'] = text
    neg['label'] = 0
    neg['augmented'] = True
    neg['neg_type'] = 'defused_joke'
    neg['rationale'] = "Modified joke where the humorous element has been neutralized."
    return neg


def generate_wrong_topic_joke(item, model):
    """Type 3: A real joke but on a completely different topic. Teaches query relevance."""
    prompt = f"""Write a short, original one-liner joke or pun about a COMPLETELY DIFFERENT topic than "{item.get('query', 'the original topic')}".
The joke should be genuinely funny but have NOTHING to do with the original query.
Output ONLY the joke, nothing else.

Original topic query: "{item.get('query', 'General')}"
Unrelated joke:"""

    text = ollama_generate(prompt, model)
    if not text or len(text) < 10:
        return None

    neg = copy.deepcopy(item)
    neg['text'] = text
    neg['label'] = 0
    neg['augmented'] = True
    neg['neg_type'] = 'wrong_topic_joke'
    neg['rationale'] = "A genuine joke but completely unrelated to the query topic."
    return neg


def main():
    """
    Step 2: Generate hard negatives for cross-encoder training.
    Creates THREE types of hard negatives for maximum discrimination:
      1. Literal rewrites  - same topic, no humor (teaches "relevant != funny")
      2. Defused jokes      - humor stripped, structure kept (teaches subtle difference)
      3. Wrong-topic jokes  - funny but irrelevant to query (teaches "funny != relevant")

    Supports resuming from where it left off if interrupted.

    Usage: python step2_old_hard_negatives.py [--model gemma4:e4b] [--workers 4]
    """
    print(f"Step 2: Generating Hard Negatives with {args.model} ({args.workers} workers)\n")
    start_time = time.time()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    positives = [item for item in data if item.get('label') == 1]
    print(f"\tTotal items: {len(data)}")
    print(f"\tPositive items (will generate negatives from): {len(positives)}")

    existing_augmented = set()
    all_results = list(data)

    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for item in existing:
                if item.get('augmented'):
                    key = (item.get('text', ''), item.get('neg_type', ''))
                    existing_augmented.add(key)
                    all_results.append(item)
            print(f"\tAlready generated negatives: {len(existing_augmented)} (will skip)")
        except (json.JSONDecodeError, Exception):
            pass

    tasks = []
    for item in positives:
        tasks.append(('literal_rewrite', item, generate_literal_rewrite))
        if not args.skip_defused:
            tasks.append(('defused_joke', item, generate_defused_joke))
        if not args.skip_wrong_topic:
            tasks.append(('wrong_topic_joke', item, generate_wrong_topic_joke))

    original_count = len(tasks)
    print(f"\tTasks to process: {len(tasks)}")

    completed = 0
    generated = 0
    errors = 0

    def process_task(task_tuple):
        neg_type, item, gen_func = task_tuple
        result = gen_func(item, args.model)
        return neg_type, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_task, t): t for t in tasks}

        for future in as_completed(futures):
            completed += 1
            try:
                neg_type, result = future.result()
                if result is not None:
                    all_results.append(result)
                    generated += 1
            except Exception as e:
                errors += 1

            if completed % 50 == 0 or completed == len(tasks):
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)

                elapsed = time.time() - start_time
                avg = elapsed / completed if completed else 1
                eta = avg * (len(tasks) - completed)
                print(f"\t[{completed}/{len(tasks)}] "
                      f"Generated: {generated} | Errors: {errors} | "
                      f"ETA: {datetime.timedelta(seconds=int(eta))}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    type_counts = {}
    for item in all_results:
        if item.get('augmented'):
            t = item.get('neg_type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nStep 2 Complete!")
    print(f"\tOriginal items:     {len(data)}")
    print(f"\tGenerated negatives: {generated}")
    for neg_type, count in sorted(type_counts.items()):
        print(f"\t\t- {neg_type}: {count}")
    print(f"\tTotal dataset size: {len(all_results)}")
    print(f"\tTime: {datetime.timedelta(seconds=int(time.time() - start_time))}")
    print(f"\tOutput: {args.output}")


if __name__ == "__main__":
    main()

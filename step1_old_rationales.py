import argparse
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
parser.add_argument('--input', default=os.path.join(PathManager.DATA_DIR, 'processed_joker_train.json'))
parser.add_argument('--output', default=os.path.join(PathManager.DATA_DIR, 'temp_step1_rationales.json'))
args = parser.parse_args()

os.makedirs('data', exist_ok=True)


def load_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def load_progress(output_path):
    """Load already-processed items to support resume."""
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        done = {}
        for item in existing:
            key = (item.get('text', ''), item.get('query', ''))
            done[key] = item
        return done
    except (json.JSONDecodeError, Exception):
        return {}


def generate_rationale(item, model):
    """Generate a single rationale via Ollama."""
    text = item['text']
    query = item.get('query', 'General Wordplay')
    is_joke = item['label'] == 1
    joke_status = "IS a relevant pun/joke" if is_joke else "is NOT a relevant pun/joke"

    prompt = f"""Analyze the following text based on the search query "{query}".
Write EXACTLY ONE sentence explaining WHY it {joke_status}. 
- If it is a joke, identify the specific linguistic mechanism (e.g., the exact pun, double meaning, or misdirection used). 
- If it is not a joke, explain why it is merely a literal or unrelated statement.

Text: "{text}"
One-Sentence Rationale:"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'num_gpu': 99}
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"\n[!] Rationale Error: {e}")
        return f"The text exhibits linguistic features aligned with the '{joke_status}' classification."


def main():
    print(f"Step 1: Generating Rationales with {args.model} ({args.workers} workers)\n")
    start_time = time.time()

    data = load_input(args.input)
    done = load_progress(args.output)
    total = len(data)

    print(f"\tTotal items: {total}")
    print(f"\tAlready done: {len(done)} (will skip)")

    results = list(done.values())
    pending = []

    for item in data:
        item['query'] = item.get('query', 'General Wordplay')
        item['augmented'] = False
        key = (item.get('text', ''), item.get('query', ''))
        if key not in done:
            pending.append(item)

    print(f"\tRemaining: {len(pending)}\n")

    if not pending:
        print("Done. All items already processed. Nothing to do.")
        return

    completed = 0
    errors = 0

    def process_item(item):
        rationale = generate_rationale(item, args.model)
        item['rationale'] = rationale
        return item

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_item, item): item for item in pending}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
            except Exception as e:
                errors += 1
                item = futures[future]
                item['rationale'] = "Error generating rationale."
                results.append(item)
                completed += 1

            if completed % 50 == 0 or completed == len(pending):
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

                elapsed = time.time() - start_time
                avg = elapsed / completed
                eta = avg * (len(pending) - completed)
                print(
                    f"\t[{completed}/{len(pending)}] "
                    f"Elapsed: {datetime.timedelta(seconds=int(elapsed))} | "
                    f"ETA: {datetime.timedelta(seconds=int(eta))} | "
                    f"Errors: {errors}"
                )

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nStep 1 Complete! {completed} rationales in {datetime.timedelta(seconds=int(time.time() - start_time))}.")
    print(f"\tOutput: {args.output}")


if __name__ == "__main__":
    main()

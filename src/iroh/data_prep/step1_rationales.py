import argparse
import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama

from iroh.core.path_manager import PathManager


def parse_args():
    parser = argparse.ArgumentParser(description="Generate humor rationales with Ollama")
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model name")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Ollama workers")
    parser.add_argument("--input", default=os.path.join(PathManager.DATA_DIR, "processed_joker_train.json"))
    parser.add_argument("--output", default=os.path.join(PathManager.DATA_DIR, "temp_step1_rationales.json"))
    parser.add_argument(
        "--qrels", default=os.path.join(PathManager.DATA_DIR, "joker_task1_retrieval_qrels_train26_english.json")
    )
    parser.add_argument(
        "--queries", default=os.path.join(PathManager.DATA_DIR, "joker_task1_retrieval_queries_test26_english.json")
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def build_docid_query_map(qrels_path, queries_path):
    """Map docids to their actual queries from qrels."""
    mapping = {}
    try:
        qrels = load_json(qrels_path)
        queries = load_json(queries_path)
        qid_to_query = {str(q["qid"]): q["query"] for q in queries}
        for item in qrels:
            docid = str(item.get("docid", ""))
            qid = str(item.get("qid", ""))
            if qid in qid_to_query:
                mapping.setdefault(docid, []).append(qid_to_query[qid])
    except FileNotFoundError:
        print("\t\t[!] Qrels/queries not found, using 'General Wordplay' as default query")
    return mapping


def load_progress(output_path):
    if not os.path.exists(output_path):
        return {}
    try:
        existing = load_json(output_path)
        return {(item.get("text", ""), item.get("query", "")): item for item in existing}
    except Exception:
        return {}


def generate_rationale(item, model):
    text = item["text"]
    query = item.get("query", "General Wordplay")
    is_joke = item["label"] == 1
    joke_status = "IS a relevant pun/joke" if is_joke else "is NOT a relevant pun/joke"

    prompt = f"""Analyze the following text in the context of the search query "{query}".
Write EXACTLY ONE concise sentence explaining WHY this text {joke_status}.

If it IS a joke/pun, identify the SPECIFIC linguistic mechanism:
- Homophonic pun (words that sound alike but differ in meaning)
- Homographic pun (same spelling, different meanings)
- Compound pun (multiple puns in one text)
- Tom Swifty (adverb that creates a pun with the dialogue)
- Double entendre / double meaning
- Malapropism / word substitution
- Ironic twist / misdirection

If it is NOT a joke, explain: is it factual? definitions? unrelated topic? lacks wordplay?

Text: "{text}"
One-Sentence Rationale:"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_gpu": 99},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"\n[!] Error: {e}")
        return f"Classification: {joke_status}."


def main():
    """
    Generates one-sentence humor rationales for training examples using a local LLM (default: gemma4:e4b via Ollama).

    The rationale identifies the specific linguistic mechanism (homophonic pun, Tom Swifty, double entendre, etc.)
    or explains why a text is NOT humorous.

    Usage:
        python step1_rationales.py [--model gemma4:e4b] [--workers 4]
        python step1_rationales.py --input data/processed_joker_train.json
    """
    args = parse_args()

    print(f"Step 1: Generating Rationales with {args.model} ({args.workers} workers)\n")
    start_time = time.time()

    data = load_json(args.input)

    docid_query_map = build_docid_query_map(args.qrels, args.queries)
    print(f"\tMapped {len(docid_query_map)} docids to queries")

    for item in data:
        docid = item.get("docid", "")
        if docid in docid_query_map:
            item["query"] = docid_query_map[docid][0]
        elif "query" not in item:
            item["query"] = "General Wordplay"
        item["augmented"] = False

    done = load_progress(args.output)
    print(f"\tTotal items: {len(data)}")
    print(f"\tAlready done: {len(done)} (will skip)")

    results = list(done.values())
    pending = [item for item in data
               if (item.get("text", ""), item.get("query", "")) not in done]

    print(f"\tRemaining: {len(pending)}\n")

    if not pending:
        print("All items already processed.")
        return

    completed = 0
    errors = 0

    def process_item(item):
        rationale = generate_rationale(item, args.model)
        item["rationale"] = rationale
        return item

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

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
                item["rationale"] = "Error generating rationale."
                results.append(item)
                completed += 1

            if completed % 25 == 0 or completed == len(pending):
                with open(args.output, "w", encoding="utf-8") as f:
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

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nStep 1 Complete! {completed} rationales in {datetime.timedelta(seconds=int(time.time() - start_time))}.")
    print(f"\tOutput: {args.output}")


if __name__ == "__main__":
    main()

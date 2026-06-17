import json
import os
import sys

from iroh.core.path_manager import PathManager

FILE_PAIRS = [
    ("temp_step1_rationales.json", "temp_step1_rationales.json"),
    ("temp_step2_augmented.json", "temp_step2_augmented.json"),
    ("old_temp_step1_rationales.json", "old_temp_step1_rationales.json"),
    ("old_temp_step2_augmented.json", "old_temp_step2_augmented.json"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved {len(data)} entries → {path}")


def normalize_text(text):
    return " ".join(text.strip().split())


def compare_entries(entry_a, entry_b):
    """Return a dict of fields that differ between two entries (excluding 'text')."""
    diffs = {}
    all_keys = set(entry_a.keys()) | set(entry_b.keys())
    for key in sorted(all_keys):
        if key == "text":
            continue
        val_a = entry_a.get(key, "<MISSING>")
        val_b = entry_b.get(key, "<MISSING>")
        if val_a != val_b:
            diffs[key] = {"data": val_a, "old": val_b}
    return diffs


def process_pair(data_path, old_path, output_path):
    print(f"\n{'═' * 70}")
    print(f"  DATA: {data_path}")
    print(f"  OLD:  {old_path}")
    print(f"{'═' * 70}")

    if not os.path.exists(data_path):
        print(f"\t[!] Data file not found, skipping.")
        return
    if not os.path.exists(old_path):
        print(f"\t[!]Old file not found, skipping.")
        return

    data_entries = load_json(data_path)
    old_entries = load_json(old_path)

    print(f"\tEntries in data/: {len(data_entries)}")
    print(f"\tEntries in old/:  {len(old_entries)}")

    old_by_text = {}
    for entry in old_entries:
        key = normalize_text(entry.get("text", ""))
        old_by_text.setdefault(key, []).append(entry)

    data_by_text = {}
    for entry in data_entries:
        key = normalize_text(entry.get("text", ""))
        data_by_text.setdefault(key, []).append(entry)

    exact_duplicates = 0
    text_dupes_diff_fields = 0
    only_in_data = 0
    only_in_old = 0
    field_diff_details = []

    shared_texts = set(data_by_text.keys()) & set(old_by_text.keys())
    data_only_texts = set(data_by_text.keys()) - set(old_by_text.keys())
    old_only_texts = set(old_by_text.keys()) - set(data_by_text.keys())

    only_in_data = sum(len(data_by_text[t]) for t in data_only_texts)
    only_in_old = sum(len(old_by_text[t]) for t in old_only_texts)

    for txt in shared_texts:
        for d_entry in data_by_text[txt]:
            matched = False
            for o_entry in old_by_text[txt]:
                diffs = compare_entries(d_entry, o_entry)
                if not diffs:
                    exact_duplicates += 1
                    matched = True
                    break
            if not matched:
                text_dupes_diff_fields += 1
                diffs = compare_entries(d_entry, old_by_text[txt][0])
                snippet = d_entry.get("text", "")[:80]
                field_diff_details.append((snippet, diffs))

    print(f"\n  -- Results --")
    print(f"\tExact duplicates (text + all fields):  {exact_duplicates}")
    print(f"\tSame text, different other fields:     {text_dupes_diff_fields}")
    print(f"\tOnly in data/:                         {only_in_data}")
    print(f"\tOnly in old/:                          {only_in_old}")

    if field_diff_details:
        show = min(len(field_diff_details), 5)
        print(f"\n  -- Field differences (showing {show}/{len(field_diff_details)}) --")
        for snippet, diffs in field_diff_details[:show]:
            print(f'  Text: "{snippet}..."')
            for field, vals in diffs.items():
                print(f"    {field}: data={vals['data']!r}  |  old={vals['old']!r}")
            print()

    merged = {}
    for entry in old_entries:
        key = normalize_text(entry.get("text", ""))
        merged[key] = entry
    for entry in data_entries:
        key = normalize_text(entry.get("text", ""))
        merged[key] = entry

    merged_list = list(merged.values())
    print(f"\n\tMerged total (deduplicated): {len(merged_list)}")
    save_json(merged_list, output_path)


def main():
    """
    Compare paired JSON files from data/ and old/ directories:
        1. Check for duplicates based on 'text' field.
        2. For text duplicates, compare other fields.
        3. Write merged (deduplicated) output to a new JSON file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        base_dir = sys.argv[1]

    data_dir = PathManager.DATA_DIR
    old_dir = os.path.join(data_dir, "old")
    output_dir = os.path.join(data_dir, "merged")

    print(f"Base directory: {base_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Old dir:  {old_dir}")
    print(f"Output:   {output_dir}")

    for data_file, old_file in FILE_PAIRS:
        data_path = os.path.join(data_dir, data_file)
        old_path = os.path.join(old_dir, old_file)
        out_name = f"merged_{data_file}"
        output_path = os.path.join(output_dir, out_name)

        process_pair(data_path, old_path, output_path)

    print(f"\n{'═' * 70}")
    print("Done! Merged files are in the 'merged/' directory.")


if __name__ == "__main__":
    main()

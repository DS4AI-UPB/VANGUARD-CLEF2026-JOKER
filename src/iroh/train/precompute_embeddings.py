import argparse
import os
import sys
import time

import torch

from iroh.core.config import CORPUS_EMB_CACHE, DENSE_MODEL, ensure_dirs
from iroh.core.path_manager import PathManager
from iroh.core.utils import load_corpus, seed_everything


def main():
    """
    Run this before pipeline.py to cache the dense embeddings on CPU.
    Tries to avoid OOM crashes when the CE model is also loaded on a small GPU.

    Usage:
        python precompute_embeddings.py
        python precompute_embeddings.py --device cpu      # force CPU
        python precompute_embeddings.py --batch-size 64   # smaller batches if OOM
        python precompute_embeddings.py --force            # recompute even if cached
    """
    parser = argparse.ArgumentParser(description="Pre-compute corpus embeddings")
    parser.add_argument("--device", type=str, default=None, help="Force device: 'cpu' or 'cuda' (default: auto)")
    parser.add_argument("--batch-size", type=int, default=128, help="Encoding batch size (reduce if OOM)")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    seed_everything()
    ensure_dirs()

    if os.path.exists(CORPUS_EMB_CACHE) and not args.force:
        try:
            existing = torch.load(CORPUS_EMB_CACHE, weights_only=True, map_location="cpu")
            size_mb = existing.element_size() * existing.nelement() / (1024 ** 2)
            print(f"Cache already exists: {CORPUS_EMB_CACHE}")
            print(f"\tShape: {existing.shape}, Size: {size_mb:.1f} MB")
            print(f"Use --force to recompute.")
            return
        except Exception as e:
            print(f"Existing cache is corrupted ({e}), recomputing...")

    if not os.path.exists(PathManager.CORPUS_FILE):
        print(f"ERROR: Corpus file not found: {PathManager.CORPUS_FILE}")
        print(f"\tExpected in: {PathManager.DATA_DIR}")
        sys.exit(1)

    print(f"Loading corpus from {PathManager.CORPUS_FILE}...")
    try:
        _, corpus_texts, _ = load_corpus(PathManager.CORPUS_FILE)
    except Exception as e:
        print(f"ERROR: Failed to load corpus: {e}")
        sys.exit(1)

    if not corpus_texts:
        print("ERROR: Corpus is empty!")
        sys.exit(1)

    original_len = len(corpus_texts)
    corpus_texts = [t for t in corpus_texts if t and t.strip()]
    if len(corpus_texts) < original_len:
        print(f"\tFiltered {original_len - len(corpus_texts)} empty documents")
    print(f"\t{len(corpus_texts)} documents to encode")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\tGPU: {name} ({mem_gb:.0f} GB)")
    else:
        device = "cpu"
        print("\tNo GPU - using CPU (slower but works if enough ram)")

    print(f"\nLoading embedder: {DENSE_MODEL}")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(DENSE_MODEL, device=device)
    except Exception as e:
        print(f"ERROR: Failed to load SentenceTransformer: {e}")
        print(f"\tTry: pip install sentence-transformers --break-system-packages")
        sys.exit(1)

    batch_size = args.batch_size
    print(f"\nEncoding {len(corpus_texts)} documents (batch_size={batch_size}, device={device})...")
    start_time = time.time()

    try:
        embeddings = embedder.encode(
            corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size, device=device
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device == "cuda" and ("out of memory" in str(e).lower() or "CUDA" in str(e)):
            print(f"\n\tGPU OOM with batch_size={batch_size}, retrying with 32...")
            torch.cuda.empty_cache()
            try:
                embeddings = embedder.encode(
                    corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=32, device=device
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                print(f"\tStill OOM. Falling back to CPU...")
                torch.cuda.empty_cache()
                embedder = SentenceTransformer(DENSE_MODEL, device="cpu")
                embeddings = embedder.encode(
                    corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size
                )
        else:
            raise

    elapsed = time.time() - start_time
    print(f"\tEncoding done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    embeddings = embeddings.cpu()
    os.makedirs(os.path.dirname(CORPUS_EMB_CACHE), exist_ok=True)
    torch.save(embeddings, CORPUS_EMB_CACHE)

    size_mb = embeddings.element_size() * embeddings.nelement() / (1024 ** 2)
    print(f"\nSaved: {CORPUS_EMB_CACHE}")
    print(f"\tShape: {embeddings.shape}, Size: {size_mb:.1f} MB")

    try:
        verify = torch.load(CORPUS_EMB_CACHE, weights_only=True, map_location="cpu")
        assert verify.shape == embeddings.shape
        print(f"\tVerified: cache loads correctly")
    except Exception as e:
        print(f"\tWARNING: verification failed: {e}")

    print(f"\nDone... pipeline.py will load from cache automatically.")


if __name__ == "__main__":
    main()

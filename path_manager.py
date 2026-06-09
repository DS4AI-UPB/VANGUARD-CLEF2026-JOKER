import os
from pathlib import Path


class PathManager:
    BASE = Path(__file__).resolve().parent
    DATA_DIR = os.path.join(BASE, "data")
    MODELS_DIR = os.path.join(BASE, "models")
    RESULTS_DIR = os.path.join(BASE, "results")
    PLOT_DIR = os.path.join(BASE, "plots")

    CORPUS_FILE = os.path.join(DATA_DIR, "joker_task1_retrieval_corpus26_english.json")
    CORPUS_2025_FILE = os.path.join(DATA_DIR, "joker_task1_retrieval_corpus25_EN.json")
    QUERIES_TRAIN_FILE = os.path.join(DATA_DIR, "joker_task1_retrieval_queries_train26_english.json")
    QUERIES_TRAIN_2025_FILE = os.path.join(DATA_DIR, "joker_task1_retrieval_queries_test25_EN.json")
    QRELS_TRAIN_FILE = os.path.join(DATA_DIR, "joker_task1_retrieval_qrels_train26_english.json")
    QRELS_TRAIN_2025_FILE = os.path.join(DATA_DIR, "joker_task1_retrieval_qrels_train25_EN.json")
    QUERIES_TEST_FILE = os.path.join(DATA_DIR, "joker_task1_retrieval_queries_test26_english.json")
    QUERIES_TEST_2025_FILE = os.path.join(DATA_DIR, "joker_task1_retrieval_queries_test25_EN.json")

    PROCESSED_TRAIN_FILE = os.path.join(DATA_DIR, "processed_joker_train.json")
    LOCAL_TRAIN_QUERIES = os.path.join(DATA_DIR, "local_train_queries.json")
    LOCAL_TEST_QUERIES = os.path.join(DATA_DIR, "local_test_queries.json")
    LOCAL_TEST_QRELS = os.path.join(DATA_DIR, "local_test_qrels.json")
    CORPUS_HUMOR_CACHE = os.path.join(DATA_DIR, "corpus_humor_scores.json")

    RATIONALES_FILE = os.path.join(DATA_DIR, "temp_step1_rationales.json")
    AUGMENTED_FILE = os.path.join(DATA_DIR, "temp_step2_augmented.json")
    OLD_RATIONALES_FILE = os.path.join(DATA_DIR, "old_temp_step1_rationales.json")
    OLD_AUGMENTED_FILE = os.path.join(DATA_DIR, "old_temp_step2_augmented.json")

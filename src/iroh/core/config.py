import os

from iroh.core.path_manager import PathManager

SEED = 42

DENSE_MODEL = "BAAI/bge-base-en-v1.5"

CE_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BGE_BASE_MODEL = "BAAI/bge-reranker-base"
CE_GTE_BASE_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"

CE_CANDIDATES = [
    "CE_GTE_new",
    "CE_GTE_new_aug",
    "CE_BGE_new",
    "CE_BGE_new_aug",
    "CE_MiniLM_new",
    "CE_MiniLM_new_aug",
]

JUDGE_BASE_MODEL = "google/gemma-4-31B-it"
JUDGE_LORA_R = 32
JUDGE_LORA_ALPHA = 64
JUDGE_MAX_LENGTH = 384
JUDGE_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

JUDGE_CANDIDATES = [
    "Judge_Qwen7B_new",
    "Judge_Qwen7B_new_aug",
    "Judge_Qwen7B_old",
    "Judge_Qwen7B_old_aug",
    "Judge_G4_31B_new",
    "Judge_G4_31B_new_aug",
    "Judge_G4_31B_old",
    "Judge_G4_31B_old_aug",
]

JUDGE_SYSTEM_PROMPT = (
    "You are a humor and wordplay detection judge. You evaluate whether "
    "a text is relevant to a query AND contains humor, jokes, puns, wordplay, "
    "or any form of linguistic wit (double meanings, homophones, malapropisms, "
    "ironic twists). Answer only YES or NO."
)

STAGE1_TOPK = 2000
USE_EXPANDED_BM25 = True
RRF_K = 60
RRF_BM25_WEIGHT = 1.0
RRF_DENSE_WEIGHT = 1.0
RRF_HUMOR_BOOST = 0
HUMOR_PRIOR_MIN = 0.60
BM25_K1 = 1.2
BM25_B = 0.5
QUERY_EXPANSION_VARIANTS = 3
STAGE2_TOPK = 500
STAGE2_CE_RANK_BLEND = 0.85
STAGE2_DEDUP_THRESHOLD = 0.97

STAGE2_CE_BATCH_SIZE = 256
STAGE3_TOPK = 1000
CE_WEIGHT = 0.50
JUDGE_WEIGHT = 0.50
JUDGE_THRESHOLD = 0.40

_CE_DEFAULTS = {
    "batch_size": 128,
    "max_epochs": 50,
    "patience": 3,
    "lr": 1e-5,
    "warmup_ratio": 0.15,
    "weight_decay": 0.02,
    "base_model": CE_BASE_MODEL,
    "ce_batch_size_infer": 256,
    "automodel_args": None,
}

_CE_BGE_DEFAULTS = {
    "batch_size": 32,
    "max_epochs": 50,
    "patience": 3,
    "lr": 1e-5,
    "warmup_ratio": 0.15,
    "weight_decay": 0.02,
    "base_model": CE_BGE_BASE_MODEL,
    "ce_batch_size_infer": 64,
    "automodel_args": None,
}

_CE_GTE_DEFAULTS = {
    "batch_size": 8,
    "max_epochs": 50,
    "patience": 3,
    "lr": 2e-5,
    "warmup_ratio": 0.15,
    "weight_decay": 0.02,
    "base_model": CE_GTE_BASE_MODEL,
    "ce_batch_size_infer": 16,
    "automodel_args": {"torch_dtype": "auto", "attn_implementation": "eager"},
}

CE_TRAIN_CONFIGS = [
    {**_CE_DEFAULTS, "name": "CE_MiniLM_new",
     "rationale_file": PathManager.RATIONALES_FILE, "augmented_file": PathManager.AUGMENTED_FILE,
     "use_augmented": False},
    {**_CE_DEFAULTS, "name": "CE_MiniLM_new_aug",
     "rationale_file": PathManager.RATIONALES_FILE, "augmented_file": PathManager.AUGMENTED_FILE,
     "use_augmented": True},

    {**_CE_BGE_DEFAULTS, "name": "CE_BGE_new",
     "rationale_file": PathManager.RATIONALES_FILE, "augmented_file": PathManager.AUGMENTED_FILE,
     "use_augmented": False},
    {**_CE_BGE_DEFAULTS, "name": "CE_BGE_new_aug",
     "rationale_file": PathManager.RATIONALES_FILE, "augmented_file": PathManager.AUGMENTED_FILE,
     "use_augmented": True},

    {**_CE_GTE_DEFAULTS, "name": "CE_GTE_new",
     "rationale_file": PathManager.RATIONALES_FILE, "augmented_file": PathManager.AUGMENTED_FILE,
     "use_augmented": False},
    {**_CE_GTE_DEFAULTS, "name": "CE_GTE_new_aug",
     "rationale_file": PathManager.RATIONALES_FILE, "augmented_file": PathManager.AUGMENTED_FILE,
     "use_augmented": True},
]


def get_ce_train_config(name: str) -> dict | None:
    """Look up a CE training config by name. Returns None if not found."""
    for c in CE_TRAIN_CONFIGS:
        if c["name"] == name:
            return c
    return None


def get_ce_infer_batch_size(ce_name: str, default: int = 256) -> int:
    """
    Pipeline-side: pick a safe Stage-2 inference batch size for a given CE model name.
    Falls back to name-based heuristics for models not in CE_TRAIN_CONFIGS (e.g. legacy checkpoints).
    """
    cfg = get_ce_train_config(ce_name)
    if cfg is not None:
        return int(cfg.get("ce_batch_size_infer", default))
    upper = ce_name.upper()
    if "GTE" in upper:
        return 16
    if "BGE" in upper:
        return 64
    return default


def get_ce_automodel_args(ce_name: str) -> dict | None:
    """
    Return any extra kwargs that should be passed to CrossEncoder's automodel_args parameter.
    GTE needs torch_dtype="auto" while others don't.
    """
    cfg = get_ce_train_config(ce_name)
    if cfg is not None:
        return cfg.get("automodel_args", None)
    if "GTE" in ce_name.upper():
        return {"torch_dtype": "auto", "attn_implementation": "eager"}
    return None


_JUDGE_DEFAULTS_G4_31B = {
    "model_id": "google/gemma-4-31B-it",
    "lora_r": 32, "lora_alpha": 64,
    "max_epochs": 30, "patience": 2, "grad_accum": 4,
    "lr": 5e-5, "max_length": 384,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
}

_JUDGE_DEFAULTS_QWEN7B = {
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "lora_r": 64, "lora_alpha": 128,
    "max_epochs": 30, "patience": 2, "grad_accum": 8,
    "lr": 2e-4, "max_length": 384,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
}


def _build_judge_data_variants(defaults, prefix):
    """Build the 4 data-variant configs for a given model (Table 4 in paper)."""
    return [
        {**defaults, "name": f"{prefix}_new",
         "rationale_file": PathManager.RATIONALES_FILE, "augmented_file": PathManager.AUGMENTED_FILE,
         "use_augmented": False},
        {**defaults, "name": f"{prefix}_new_aug",
         "rationale_file": PathManager.RATIONALES_FILE, "augmented_file": PathManager.AUGMENTED_FILE,
         "use_augmented": True},
        {**defaults, "name": f"{prefix}_old",
         "rationale_file": PathManager.OLD_RATIONALES_FILE, "augmented_file": PathManager.OLD_AUGMENTED_FILE,
         "use_augmented": False},
        {**defaults, "name": f"{prefix}_old_aug",
         "rationale_file": PathManager.OLD_RATIONALES_FILE, "augmented_file": PathManager.OLD_AUGMENTED_FILE,
         "use_augmented": True},
    ]


JUDGE_GEMMA_31B_CONFIGS = _build_judge_data_variants(_JUDGE_DEFAULTS_G4_31B, "Judge_G4_31B")
JUDGE_QWEN7B_CONFIGS = _build_judge_data_variants(_JUDGE_DEFAULTS_QWEN7B, "Judge_Qwen7B")

JUDGE_TRAIN_CONFIGS = JUDGE_QWEN7B_CONFIGS + JUDGE_GEMMA_31B_CONFIGS

EVAL_METRICS = {"map", "recall_30", "recall_1000", "ndcg_cut_10", "P_5", "P_10"}

CORPUS_EMB_CACHE = os.path.join(PathManager.DATA_DIR, "corpus_emb_bge-base.pt")

CORPUS_EMB_CACHE_LARGE = os.path.join(PathManager.DATA_DIR, "corpus_emb_bge-large.pt")
CORPUS_EMB_CACHE_M3 = os.path.join(PathManager.DATA_DIR, "corpus_emb_bge-m3.pt")
CORPUS_EMB_CACHE_ICL = os.path.join(PathManager.DATA_DIR, "corpus_emb_bge-en-icl.pt")

EMBEDDER_CANDIDATES = [
    {
        "name": "bge-base",
        "model_id": "BAAI/bge-base-en-v1.5",
        "cache_file": CORPUS_EMB_CACHE,
        "icl": False,
        "description": "Baseline (768-dim, 512-tok)",
    },
    {
        "name": "bge-large",
        "model_id": "BAAI/bge-large-en-v1.5",
        "cache_file": CORPUS_EMB_CACHE_LARGE,
        "icl": False,
        "description": "Larger same-family model (1024-dim)",
    },
    {
        "name": "bge-m3",
        "model_id": "BAAI/bge-m3",
        "cache_file": CORPUS_EMB_CACHE_M3,
        "icl": False,
        "description": "Multi-granularity: dense + sparse + ColBERT (1024-dim)",
    },
    {
        "name": "bge-en-icl",
        "model_id": "BAAI/bge-en-icl",
        "cache_file": CORPUS_EMB_CACHE_ICL,
        "icl": True,
        "description": "In-context learning embedder - query prefix encodes task (4096-dim, 7B)",
    },
]
HUMOR_BATCH_SAVE_EVERY = 100
STOP_WORDS = frozenset({
    "what", "how", "why", "the", "a", "an", "is", "are", "was",
    "were", "do", "does", "did", "can", "could", "about", "with",
    "for", "and", "or", "in", "on", "at", "to", "of", "that", "this",
})


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [PathManager.DATA_DIR, PathManager.MODELS_DIR, PathManager.RESULTS_DIR, PathManager.PLOT_DIR]:
        os.makedirs(d, exist_ok=True)

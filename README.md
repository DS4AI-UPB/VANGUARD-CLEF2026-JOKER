# IROH - Presentation Website

[![Paper](https://img.shields.io/badge/Paper-CLEF%20JOKER%202026-7c3aed)](https://When-Paper-Appears-it-Will-Work.com)
[![Code Implementation](https://img.shields.io/badge/Code-Implementation-green)](https://github.com/DS4AI-UPB/VANGUARD-CLEF2026-JOKER)
[![arXiv](https://img.shields.io/badge/arXiv-WIP-b31b1b.svg)](https://arxiv.org/abs/WIP)
[![Leaderboard](https://img.shields.io/badge/JOKER%202026%20Task%201%20EN-1st%20%C2%B7%200.6347%20MAP-b8860b)](https://github.com/DS4AI-UPB/VANGUARD-CLEF2026-JOKER)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the source code and assets for the official project website of the paper
**"IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English"**,
to be presented at the **JOKER Lab @ CLEF 2026** by team **VANGUARD**.

**Live Website:** [ds4ai-upb.github.io/VANGUARD-CLEF2026-JOKER](https://ds4ai-upb.github.io/VANGUARD-CLEF2026-JOKER/)

## Paper Summary

**Authors:** Ana-Maria Luisa Mocanu, Sebastian Mocanu, Ciprian-Octavian Truică, Elena-Simona Apostol

**Abstract:**

> Our team, VANGUARD, presents IROH (Insightful Ranking of Humor), a three-stage retrieval system for JOKER Task~1 English at CLEF~2026, achieving first place on the leaderboard with 0.6347 MAP. Our pipeline combines hybrid sparse-dense retrieval, cross-encoder reranking, and a LoRA-adapted Large Language Model judge ensemble. We employ Gemma~4 to generate query-aware rationales under two prompt strategies, generic and typed, and produce up to four types of structured hard negatives for training data construction. Through an ablation across three cross-encoder architectures, four dense embedders, and eight judge configurations, our key findings are threefold: (1) the rationale-distilled judge is the primary driver of ranking quality, whereas appending rationales to the first-stage index contributes negligibly; (2) structured hard negatives degrade generalisation in nearly all configurations despite inflating local validation scores; and (3) across the components we ablate, the lighter, better-calibrated model is competitive with or stronger than its larger counterpart, with the generic-rationale Qwen2.5-7B judge (0.6055 MAP) outperforming every Gemma-4-31B configuration, and the advantage of generic over typed rationales is concentrated almost entirely in the smaller model.

## Resources
- [Paper (CLEF JOKER 2026 Working Notes)](https://When-Paper-Appears-it-Will-Work.com) - placeholder until the official proceedings entry is available
- [arXiv](https://arxiv.org/abs/WIP) - WIP
- [Code](https://github.com/DS4AI-UPB/VANGUARD-CLEF2026-JOKER) - full three-stage pipeline, training, and ablation scripts

## Local Development

Simply open `index.html` in a browser, or serve with any static file server:

```bash
python -m http.server 8000
```

## Deployment

The site auto-deploys to GitHub Pages via the included workflow
[.github/workflows/github-pages.yml](.github/workflows/github-pages.yml),
which handles image compression and CSS/JS minification.

## Citation
```bibtex
@InProceedings{Ana_Maria_Luisa_Mocanu_2026_IROH_CLEF,
    author    = {Mocanu, Ana-Maria Luisa and Mocanu, Sebastian and Truică, Ciprian-Octavian and Apostol, Elena-Simona},
    title     = {IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English},
    booktitle = {Conference and Labs of the Evaluation Forum (CLEF), Joker 2026 Track Task 1 English},
    month     = {June},
    year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
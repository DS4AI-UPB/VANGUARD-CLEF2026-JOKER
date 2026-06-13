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

> We present IROH (Insightful Ranking of Humor), a three-stage retrieval system for JOKER Task 1 English at CLEF 2026, achieving first place on the leaderboard with 0.6347 MAP. The pipeline combines hybrid sparse-dense retrieval, cross-encoder reranking, and a LoRA-adapted Large Language Model judge ensemble trained by rationale distillation. Our ablation shows that the rationale-distilled judge is the primary driver of ranking quality, that structured hard negatives degrade generalisation despite inflating local validation scores, and that lighter, better-calibrated models match or beat their larger counterparts.

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
<div align="center">

# SciForma: Structure-Faithful Generation of Scientific Diagrams

<p>
  <a href="https://arxiv.org/"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?logo=arxiv&logoColor=white" height="22" /></a>
  &nbsp;
  <a href="https://huggingface.co/microsoft/SciForma"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow" height="22" /></a>
  &nbsp;
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg" height="22" /></a>
</p>

</div>

---

**SciForma** is a 9B-parameter framework for the *structure-faithful* generation of scientific methodology diagrams — pipelines, architecture overviews, and dense block layouts in which a single missing component, reversed arrow, or unreadable equation can invalidate the whole figure.

SciForma decomposes diagram quality into three independently verifiable axes — **Component**, **Arrow**, and **Text** — guided by a *structural inventory* (a checklist extracted from a reference diagram). Built on this foundation, SciForma is fine-tuned from FLUX.2-klein-base-9B in two SFT stages, then post-trained with **Multi-Dimensional Conjunctive Preference Optimization (M-DPO)**, which enforces simultaneous correctness across all axes and adaptively routes gradients toward the most deficient axis. At inference time, the same structural inventory drives a verification-gated iterative refinement loop that repairs residual structural defects.

## Highlights

- **Structural inventory.** Per-axis (Component / Arrow / Text) verification with critical/moderate error severity instead of a single holistic score.
- **M-DPO.** Multi-way Bradley–Terry objective with dimension-anchored preference construction and adaptive gradient reweighting — breaks the SFT plateau where scalar DPO / GDRO / GRPO stagnate.
- **Iterative refinement.** Critic-guided localization + closed-loop inpainting + verification-and-rollback gate; lifts SciFormaBench-2K score from 69.51 → 72.40.
- **Data & benchmark.** `SciFormaData-700K` (656K generation pairs + 70K editing triplets) and `SciFormaBench-2K` (Simple 500 / Medium 900 / Hard 600) with human-verified inventories.

## Results

On **SciFormaBench-2K** (overall structural fidelity, GPT-5.4 judge):

| Method | Average | Component | Arrow | Text |
| :--- | :---: | :---: | :---: | :---: |
| Nano Banana Pro | 81.34 | 81.10 | 83.60 | 78.70 |
| GPT-Image-2 | 85.62 | 83.34 | 89.61 | 83.53 |
| GPT-Image-1.5 | 68.96 | 75.70 | 62.50 | 68.20 |
| FLUX.2-klein-base-9B | 33.87 | 51.50 | 25.20 | 23.60 |
| **SciForma-Base (SFT)** | 67.59 | 73.52 | 64.64 | 63.84 |
| **SciForma-9B (+ M-DPO)** | 69.51 | 74.49 | 66.46 | 67.00 |
| **SciForma-9B + Edit** | **72.40** | 76.70 | 69.91 | 70.14 |

On **AIBench** (VQA-style readability), SciForma-9B reaches **70.29**, slightly edging out human-drawn originals (70.09) and widening the lead over GPT-Image-1.5 by 8.67 points, with the largest margin on Topology (+6.19). As a drop-in Visualizer inside **PaperBanana**, SciForma achieves a **30.7%** overall pairwise win rate against human-drawn originals.


## Responsible AI

SciForma is released for research purposes only and is not intended for product or service deployment. It is trained on scientific figures sourced from public arXiv preprints (2015–2025) and is therefore structurally biased away from photorealistic depictions of people, places, or events; nevertheless it can still produce structurally incorrect or misleading diagrams under certain prompts, and outputs should not be treated as authoritative descriptions of any real method, system, or scientific claim. Coverage of scientific domains is skewed toward fields with high arXiv activity (machine learning, computer vision, NLP, physics, mathematics) and toward English-language captions. Downstream users are responsible for applying additional safeguards — such as human review, content moderation, and compliance checks — before broader use, and SciForma should not be used in high-stakes or regulated domains. See the SciForma model card for details.

## Privacy

This project does not collect any usage data. For more information, see the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839).

## License

This project is released under the [MIT License](LICENSE).

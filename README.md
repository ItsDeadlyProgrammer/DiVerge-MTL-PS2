<h1 align="center">DiVerge-MTL</h1>

<p align="center">
  <b>Uncertainty-weighted multi-task learning for Hinglish meme analysis.</b><br>
  Bullying · Sentiment · Emotion · Sarcasm — from one shared representation.
</p>

<p align="center">
  <b>PS2 · Minor Project Phase 2</b> · IIIT Bhopal
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/🤗_Transformers-CLIP_·_mBERT-FFD21E?style=flat-square" alt="Transformers">
  <img src="https://img.shields.io/badge/Dataset-MultiBully-4B8BBE?style=flat-square" alt="MultiBully">
  <img src="https://img.shields.io/badge/Seeds-13_·_42_·_2026-1F1B18?style=flat-square" alt="Seeds">
  <img src="https://img.shields.io/badge/Status-final_release-success?style=flat-square" alt="Final release">
</p>

<p align="center">
  <img src="assets/final_results_summary.png" width="90%" alt="Final results: ours versus the two published baselines on held-out test data">
</p>

---

## Headline numbers

| Benchmark | Metric | Published baseline | **Ours** | Δ |
| --- | --- | --- | --- | --- |
| MultiBully · bully | accuracy | 70.91 | **73.88** | +2.97 |
| MultiBully · sentiment | accuracy | 59.80 | **60.41** | +0.61 |
| MultiBully · emotion | accuracy | 29.60 | **31.84** | +2.24 |
| MultiBully · sarcasm | accuracy | 63.59 | **65.58** | +1.99 |
| **MultiBully · average** | accuracy | 55.97 | **57.93** | **+1.95** |
| HatefulMemes (Part B) | accuracy | 77.35 | **81.41** | +4.06 |
| HatefulMemes (Part B) | AUC | 81.70 | **87.90** | +6.20 |

Baselines: MultiBully paper (single-task CLIP + FC, their own split); Kapil & Ekbal (2025) for HatefulMemes.
Our numbers are the final 3-seed ensemble scored once on a held-out test split. Full protocol and caveats below.

---

## What this is

Memes carry meaning across two channels at once: the image and the overlaid text, often Hinglish — Hindi written in
Devanagari or in Roman script, mixed with English. A meme can be playful in its text and hostile in its image, and the
harm is in the interaction, not in either half.

**DiVerge-MTL** predicts four labels for one meme with a single model:

- **Bullying** — Bully / Nonbully
- **Sentiment** — Negative / Neutral / Positive
- **Emotion** — 10 classes (Angry, Anticipation, Disgust, Fear, Happiness, Other, Ridicule, Sadness, Surprise, Trust)
- **Sarcasm** — No / Yes

The name is the design: *diverge* the shared representation into per-task branches, with the task weighting learned
rather than hand-tuned. Everything runs from **one notebook** — data download, cleaning, training, ablations,
significance testing and the comparison against the base paper.

---

## Key contributions

- **A unified 4-task multimodal model** over frozen CLIP ViT-L/14 image *and* text features, with **per-task gates**
  and **learned uncertainty weighting** (Kendall et al.) instead of fixed loss weights.
- **A cleaned MultiBully benchmark**: 5,865 annotation rows → **4,899 usable unique memes** after complete-label
  filtering, image reachability, unreadable images, byte-identical duplicates and 52 re-saved copies (pHash ≤ 4 bits
  *and* ≥ 90% text similarity).
- **A noise ceiling for the dataset itself** — duplicate memes carrying conflicting labels — which bounds how good any
  model can look on this data, and is reported next to our own scores.
- **Statistics, not point estimates**: 5-fold cross-validation, paired bootstrap with 95% CIs, and
  **Holm–Bonferroni** correction across 8 comparisons (plus a 32-cell per-task family).
- **Honest ablations, including the ones that failed**: the interaction feature, the shared gate and the
  uncertainty weighting did **not** survive significance testing, and we say so.
- **A reproduction of the base paper's comparison systems** (Tables 8 and 9) trained on our own split rather than
  copied from their paper.
- **A bounded Hinglish claim**: only 2.2% of memes contain Devanagari, and we report the interval, not just the mean.

---

## Results

### 1 · MultiBully — final model vs the paper

Final model: CLIP ViT-L/14 image + text, per-task gated fusion, uncertainty-weighted loss, inverse-frequency class
weights, retrained on **train + validation** for the validation-selected epoch count, then a **3-seed ensemble**
(seeds 13, 42, 2026). Test split scored once.

| Task | Paper (CLIP + FC) | Ours, PS2 settings | Ours, best (train-only) | **Ours, final** | Majority class | Copies agree (%) |
| --- | --- | --- | --- | --- | --- | --- |
| bully | 70.91 | 70.57 | 73.97 | **73.88** | 51.70 | 77.47 |
| sentiment | 59.80 | 57.55 | 59.37 | **60.41** | 42.59 | 69.14 |
| emotion | 29.60 | 26.53 | 29.93 | **31.84** | 15.78 | 34.57 |
| sarcasm | 63.59 | 64.40 | 65.90 | **65.58** | 60.00 | 52.16 |
| **average** | 55.97 | 54.76 | 57.29 | **57.93** | 42.52 | 58.34 |

Average **macro-F1** of the final model: **56.40** (vs 48.84 for a reproduced CLIP+FC head on ViT-B/16).

### 2 · Part B — against the base paper on its own benchmarks

Kapil & Ekbal (2025), same fusion recipe, stratified 80/20 split (seed 42); every choice made by 5-fold CV on the
development part only; the 20% test part scored once, with a **15-model ensemble** (5 folds × 3 seeds).

| Dataset | System | Test acc | Test AUC | Test wF1 |
| --- | --- | --- | --- | --- |
| HatefulMemes | Kapil & Ekbal (2025), published | 77.35 | 81.70 | — |
| HatefulMemes | **DiVerge (ours)** | **81.41** | **87.90** | **81.19** |
| MultiOFF | Kapil & Ekbal (2025), published | (79.82 F1, their split) | — | — |
| MultiOFF | **DiVerge (ours)** | **58.78** | — | **58.60** |

The base paper's exact split is not published, so Part B is an indicative comparison, not a like-for-like one.
MultiOFF's test part has 148 memes — one meme moves accuracy by 0.7 points.

### 3 · How much of the label can even be predicted? (noise ceiling)

MultiBully contains 324 memes annotated more than once. If two copies of the *same* meme disagree, no model scored
against one copy can be expected to beat that agreement rate.

| Task | Pairwise agreement (%) | Cohen's κ | Majority class (%) | Final model acc (%) | Model ÷ pairwise agreement |
| --- | --- | --- | --- | --- | --- |
| bully | 79.58 | 0.376 | 51.70 | 73.88 | 0.93 |
| sentiment | 71.67 | 0.362 | 42.59 | 60.41 | 0.84 |
| emotion | 37.50 | 0.271 | 15.78 | 31.84 | 0.85 |
| sarcasm | 56.88 | 0.149 | 60.00 | 65.58 | 1.15 |

Our model sits just under the human-consistency ceiling on three tasks and above it on sarcasm (where κ = 0.149 — the
label is close to noise). **Emotion is capped near 37%**: 10 classes with 34.6% agreement is the real explanation for
the low emotion score, not the architecture.

### 4 · Does the improvement survive correction? (paired bootstrap + Holm–Bonferroni)

Test macro-F1, 10,000 bootstrap resamples, Holm across 8 comparisons (α = 0.05).

| Comparison | Δ (F1 points) [95% CI] | p (Holm) | Survives |
| --- | --- | --- | --- |
| Final model vs PS2 settings | **+5.29 [+3.28, +7.24]** | <0.001 | ✅ |
| Best train-only config vs PS2 settings | **+4.75 [+2.94, +6.56]** | <0.001 | ✅ |
| Class weights vs none | **+1.60 [+0.54, +2.68]** | 0.0228 | ✅ |
| Final vs best train-only config | +0.54 [−0.42, +1.49] | 0.814 | ❌ |
| Per-task gates + uncertainty vs shared gate + equal weights | +0.53 [−0.26, +1.33] | 0.770 | ❌ |
| Interaction feature vs none | −0.48 [−1.45, +0.49] | 0.814 | ❌ |
| Per-task gates vs shared gate (PS features) | −0.18 [−0.91, +0.58] | 0.813 | ❌ |
| Uncertainty vs equal weights (PS features) | −0.12 [−0.29, +0.04] | 0.673 | ❌ |

**Read this table before quoting the headline.** The large, robust gains come from the *feature/optimisation* changes
(CLIP-L/14, lr 1e-4, class weights). The architectural ideas the project is named after — per-task gates and learned
σ — did not individually clear significance at this sample size. Only the **emotion** task makes the final-vs-PS gap
significant after per-task correction (Holm p < 0.001).

### 5 · 5-fold cross-validation (all 4,899 memes, same folds and seeds)

| System | Avg acc | Avg macro-F1 |
| --- | --- | --- |
| **Final config (DiVerge-MTL, CLIP-L/14)** | **57.35 ± 0.39** | **55.01 ± 0.23** |
| Best config, single seed avg | 56.71 ± 0.53 | 54.49 ± 0.57 |
| CLIP+FC (authors' head), ViT-L/14 | 55.84 ± 1.10 | 51.82 ± 1.40 |
| CLIP+FC (authors' head), ViT-B/16 | 54.35 ± 0.91 | 49.20 ± 1.15 |

Paired t-test across the 5 folds: vs ViT-B/16 head **+3.01 acc (p = 0.0012)**, **+5.81 F1 (p = 0.0002)**;
vs ViT-L/14 head **+1.52 acc (p = 0.038)**, **+3.18 F1 (p = 0.005)**.

### 6 · Ablations (best configuration, train-only)

| Variant | Avg acc | Avg F1 | Bully F1 | Sentiment F1 | Emotion F1 | Sarcasm F1 |
| --- | --- | --- | --- | --- | --- | --- |
| **Best: CLIP-L/14 + class weights, per-task gates + uncertainty** | 57.29 | **55.87** | 73.90 | 57.61 | 27.34 | 64.62 |
| same, without class weights | **57.35** | 54.26 | 73.89 | 55.89 | 24.12 | 63.13 |
| + interaction feature | 57.12 | 55.41 | 73.53 | 57.31 | 26.89 | 63.90 |
| shared gate + equal weights | 57.02 | 55.32 | 73.55 | 56.92 | 25.47 | 65.34 |
| DiVerge-MTL (PS2 settings) | 54.76 | 51.08 | 70.51 | 53.29 | 19.26 | 61.25 |

Class weights buy **+3.2 emotion F1** and rescue rare classes outright: Fear 14.88 vs 0.0 F1, Trust 27.56 vs 8.33.

A finding worth stating plainly: with the PS2 features, the **image-only probe (55.95 acc) beats the full gated
multimodal model (54.76)**. The text side only starts paying off once it is CLIP-L/14 rather than the LoRA-mBERT expert.

### 7 · The Hinglish claim, bounded

Only **108 of 4,899 memes (2.2%)** contain Devanagari, and **14 of them** are in the test split.

| Subset | Test memes | Avg acc [95% CI] | Sarcasm correct |
| --- | --- | --- | --- |
| Devanagari in text | 14 | 58.93 [42.86, 75.00] | 10/14 |
| Roman only | 721 | 57.91 [55.86, 59.95] | 472/721 |

The Devanagari subset is too small to support a robustness claim on its own. The OCR-noise augmentation + rule-based
Devanagari→Roman transliteration is in the pipeline, but its effect is measured on 14 memes — we report the interval
rather than the point estimate.

### 8 · Compute

| Configuration | Wall clock (s, 3 seeds) | Peak GPU (MB) |
| --- | --- | --- |
| Best: CLIP-L/14 + class weights, per-task gates + uncertainty | 49.7 | 304.2 |
| DiVerge-MTL (PS2 settings) | 111.5 | 159.4 |
| + interaction feature | 97.7 | 599.9 |
| Image-only probe | 143.5 | 52.1 |

Fits comfortably on a single 6 GB laptop GPU (RTX 4050) or a Colab T4.

---

<details>
<summary><b>Part D · the base paper's Tables 8 and 9, re-implemented and re-run</b> (click to expand)</summary>

Kapil & Ekbal compare against earlier systems. Part D **implements** them and trains each one on our split
(stratified 80/20, seed 42) instead of copying numbers: Stacked-LSTM / BiLSTM / CNN + VGG16, MDBERT + VGG16
(text, image, early fusion and a validation-weighted ensemble), MMBT over XLM-RoBERTa with ResNet-50 image tokens, an
average ensemble, ResNet-50 early fusion, sequential and mutual CLIP fusion, and the base paper's own hard-parameter-
sharing multi-task model with its encoder ablations.

Selected HatefulMemes results (test acc / AUC / wF1):

| System | Test acc | Test AUC | Test wF1 |
| --- | --- | --- | --- |
| Stacked-LSTM + VGG16 | 68.10 | 69.64 | 67.82 |
| MDBERT + VGG16 weighted ensemble | 76.18 | 79.24 | 75.59 |
| Mutual fusion (CLIP) | 79.92 | 86.81 | 79.73 |
| Base-paper MTL (multi-task, both datasets) | 77.33 | 84.30 | 77.39 |
| **DiVerge (ours, Part B)** | **81.41** | **87.90** | **81.19** |

<br>

<img src="assets/partD_table8_9_systems.png" width="100%" alt="Part D: every comparison system trained and scored on the same split">

Left out because they are too slow to rebuild or unavailable: DisMultiHate (needs entity extraction), the ERNIE-ViL
ensemble (PaddlePaddle only), UNITER (no official PyTorch release), and the NSFW image model. VGG16 stands in for
VGG19, XLM-RoBERTa for RoBERTa. MAMI/MMHS150K were not on disk, so Table 9 numbers published on **MAMI** are shown for
reference only — different data.

</details>

---

## Method

```
                                    ┌──────────────┐
   meme image ──▶ CLIP ViT-L/14 ───▶│              │
                                    │   gated      │──▶ head_bully     ──▶ Bully / Nonbully
   meme text  ──▶ CLIP ViT-L/14 ───▶│   fusion     │──▶ head_sentiment ──▶ Neg / Neu / Pos
        ▲                           │  h_t = g_t⊙h │──▶ head_emotion   ──▶ 10 classes
        │                           └──────────────┘──▶ head_sarcasm   ──▶ No / Yes
   Step 1: normalise + OCR-noise augment
           + Devanagari→Roman transliteration          Step 5: L = Σ 1/(2σ_t²)·L_t + log σ_t
```

| Step | What happens |
| --- | --- |
| **1 · Text preprocessing** | NFKC, zero-width removal, letter-run collapse, whitespace collapse; rule-based Devanagari→Roman Hinglish transliteration; 3 noisy OCR-style variants per training text (char dropout p=0.05, Devanagari→Roman swap p=0.5); raw text used half the time. |
| **2 · Text expert (PS2 run)** | Frozen mBERT + LoRA (r=8 on q/v of all 12 layers), CLS projected 768→512, four probe heads — **1,218,321** trainable parameters. Trained once per seed, with and without augmentation (ablation c). |
| **3 · Frozen CLIP features** | CLIP ViT-B/16 (PS2 settings) or **ViT-L/14** (best config), image and text branches, extracted once and cached to `cache/`, z-scored with training-split statistics only. |
| **4 · Gated fusion** | Per-task gate `g_t = σ(W_t h + b_t)`, `h_t = g_t ⊙ h`; variants `shared` (one gate), `image_only`, `text_only`. Heads: MLP d→256→classes. Gated model + four σ: **5,252,373** parameters. |
| **5 · Uncertainty-weighted training** | `L = Σ_t 1/(2σ_t²)·L_t + log σ_t`, AdamW, 10% linear warm-up then cosine decay, batch 32, early stopping on validation average macro-F1, inverse-frequency class weights. |
| **6 · Inference** | 3 seeds (13, 42, 2026) → for the final model, retrain each on train + validation for its validation-selected epoch count and average. |
| **7 · Ablations & statistics** | 8 comparisons, paired bootstrap (10,000 resamples), Holm–Bonferroni over two families, 5-fold CV, noise ceiling, qualitative error analysis. |

**Diagnostics from step 4/5** (final epoch, 3 seeds): learned σ stays near **0.87–0.95** for bully/sentiment/sarcasm
and **1.03** for emotion, i.e. the loss weights hover around 1 — the uncertainty weighting did *not* discover strongly
uneven task weights on this data. Mean gate values sit at **~0.50** for image vs text, so the gate does not collapse
onto one modality either. Both are reported as negative-but-informative results.

---

## Repository structure

```
DiVerge-MTL-PS2/
├── notebooks/
│   └── DiVerge_MTL_PS2_BaseModels.ipynb   # the whole project: Parts A–D, 113 cells
├── results/
│   ├── final_results_summary.csv          # headline: ours vs both baselines
│   ├── comparison.csv                     # paper / PS settings / best / final per task
│   ├── best_config_results.csv            # ablations at the best configuration
│   ├── ps_settings_results.csv            # PS2 settings + probes + σ-lr ablation
│   ├── significance.csv                   # paired bootstrap CIs
│   ├── hinglish_robustness.csv            # Devanagari vs Roman subset
│   ├── memory_time.csv                    # wall clock + peak GPU per config
│   ├── config.json                        # PS / BEST configs, cleansed counts, consistency
│   ├── test_preds_*.csv                   # per-model test predictions (12 systems)
│   ├── *.png                              # figures used in this README
│   ├── partD/                             # Tables 8+9 systems, re-implemented and scored
│   └── revision/                          # supervisor-review answers C2–C9
├── configs/
│   ├── ps2_settings.yaml                  # problem-statement configuration
│   └── best.yaml                          # improved configuration
├── docs/
│   ├── RESULTS.md                         # every table, with the exact source CSV
│   ├── ARCHITECTURE.md                     # the model, step by step
│   └── REPRODUCING.md                      # how to run it, including the smoke test
├── data/README.md                          # where to get MultiBully — data is NOT in this repo
├── assets/                                 # figures referenced by the README
├── LICENSE.md
├── CITATION.cff
└── README.md
```

**There is no data and no model weight in this repository.** MultiBully images belong to the MultiBully release; the
benchmark is referenced by filename, not copied. See [`data/README.md`](data/README.md).

---

## Running it

```bash
git clone https://github.com/ItsDeadlyProgrammer/DiVerge-MTL-PS2.git
cd DiVerge-MTL-PS2
pip install "transformers>=4.40" peft gdown imagehash openpyxl scikit-learn pandas pillow tqdm requests matplotlib
jupyter lab notebooks/DiVerge_MTL_PS2_BaseModels.ipynb
```

Everything runs from the notebook, top to bottom:

| Part | What it does | Needs |
| --- | --- | --- |
| **A** | PS2 on MultiBully — download, clean, split, 7 steps, ablations, improved config, final ensemble | MultiBully data (~5,500 images) |
| **B** | DiVerge-MTL vs the base paper on HatefulMemes / MultiOFF | `data/<Name>/frame.csv` (columns `image,text,label`) |
| **C** | Supervisor-review answers: baseline reproduction, 5-fold CV, noise ceiling, σ/gate logs, Holm correction, text-expert run, Hinglish bounds, error analysis | reuses `cache/` |
| **D** | The base paper's Tables 8 & 9 systems, implemented and trained | reuses `cache/` |

**Check the code without downloading anything:** set `SMOKE_TEST = True` in the configuration cell. The pipeline then
runs end to end in a few minutes on synthetic memes with tiny random models. It validates the code, not the results.

**Runtime:** the image download dominates. Modelling is ~30–60 min on a 6 GB GPU or a Colab T4; it runs on CPU too, much
more slowly. Every step skips work already on disk, so re-running is safe.

Full instructions and the exact reproduction order: [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

---

## Honest caveats

We would rather state these than have a reviewer find them.

1. **The architectural novelty is not statistically proven.** Per-task gates and learned σ did not clear significance
   individually (+0.53 [−0.26, +1.33] and −0.12 [−0.29, +0.04]). The gains we defend are CLIP-L/14 features, lr 1e-4,
   class weights, and train+val retraining.
2. **The full 4-task model does not beat a single-task baseline by a wide margin** in the cross-validation table, and
   at PS2 settings an image-only probe beats it outright.
3. **Emotion is near the label's noise ceiling** (37.5% pairwise agreement). Further architecture work will not move
   emotion much; better labels would.
4. **Part B comparisons are indicative**, not like-for-like: the base paper's split is unpublished and we use
   stratified 80/20.
5. **The Hinglish result rests on 14 test memes.** Treat the direction as a hypothesis, not a finding.
6. **Sarcasm exceeds its own pairwise agreement** (1.15×), which most likely means the model exploits image cues the
   annotators disagreed about — worth a dedicated study, not a victory lap.

---

## Team

**Vasudev Gupta** (23U02081) · **Harshvardhan Singh** (23U02091)
B.Tech CSE, Indian Institute of Information Technology, Bhopal

Under the supervision of **Dr. Yatendra Sahu**, Assistant Professor, Department of CSE, IIIT Bhopal.

---

## References

- Maity et al., **EMPIRICAL: Emotion Aware Multimodal Multitask Learning for Cyberbullying Detection in Code-Mixed
  Memes**, COLING 2022 — the MultiBully dataset and the PS2 setting.
- Kapil & Ekbal, **2025** — the base paper for Part B, and the source of the Tables 8/9 comparison systems.
- Kendall, Gal & Cipolla, **Multi-Task Learning Using Uncertainty to Weigh Losses**, CVPR 2018 — the σ-weighted loss.
- Radford et al., **CLIP**, ICML 2021. Kiela et al., **Supervised Multimodal Bitransformers (MMBT)**, 2019.
- Hu et al., **LoRA**, ICLR 2022.

---

## Citation

```bibtex
@misc{gupta2026diverge,
  title  = {DiVerge-MTL: Uncertainty-Weighted Multi-Task Learning for Hinglish Meme Analysis},
  author = {Vasudev Gupta and Harshvardhan Singh},
  year   = {2026},
  note   = {Minor Project Phase 2 (PS2), IIIT Bhopal},
  url    = {https://github.com/ItsDeadlyProgrammer/DiVerge-MTL-PS2}
}
```

---

## License

Academic and research use, with citation. See [`LICENSE.md`](LICENSE.md).

<p align="center"><i>If this repo or the cleaned MultiBully benchmark is useful to you, a ⭐ helps.</i></p>

# Deepfake Audio Detection: Cross-Dataset Evaluation

Evaluating how well pre-trained deepfake detectors generalize beyond their training data.

## Why This Matters

Pre-trained models often claim 99%+ accuracy, but those numbers come from evaluations on similar data. Real-world performance can be dramatically different.

## Datasets

| Dataset | Samples | Source |
|---------|---------|--------|
| [In-The-Wild](https://github.com/piotrmwojcik/In-The-Wild) | ~19k | YouTube clips |
| [Fake-or-Real](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) | ~69k | TTS-generated (Deep Voice 3, Wavenet, etc) |

## Results

| Dataset | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| Original Eval (reported) | 99.7% | - | - | - |
| In-The-Wild | 43.2% | 34.5% | 15.1% | 21.0% |
| Fake-or-Real | 59.3% | 81.5% | 24.1% | 37.2% |

The model's reported 99.7% accuracy drops to near-random on out-of-distribution data. Most concerning: ~85% of deepfakes slip through undetected (low recall), and the model remains highly confident even on wrong predictions.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_in_the_wild.ipynb` | Full evaluation on In-The-Wild dataset |
| `02_fake_or_real.ipynb` | Full evaluation on Fake-or-Real dataset |
| `03_comparison.ipynb` | Side-by-side analysis |
| `04_benchmark.ipynb` | Quick benchmark on balanced samples |

## Setup

```bash
git clone <repo-url>
cd melody
pip install -r requirements.txt
```

Download datasets to `data/` and run notebooks in order.

## Model

[MelodyMachine/Deepfake-audio-detection-V2](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) — Wav2Vec2-based classifier.

## Dependencies

- transformers, torch, librosa
- pandas, numpy, scikit-learn
- matplotlib, seaborn, tqdm

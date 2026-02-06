# Deepfake Audio Detection: Cross-Dataset Evaluation

Part of my research on audio deepfake detection, investigating how well pre-trained models generalize across different datasets.

## Motivation

Pre-trained deepfake detectors often report 99%+ accuracy, but how well do they perform on data they weren't trained on? This project evaluates a state-of-the-art model across multiple datasets to understand its real-world reliability.

## Datasets

| Dataset | Samples | Source | Description |
|---------|---------|--------|-------------|
| [In-The-Wild](https://github.com/piotrmwojcik/In-The-Wild) | 31k+ | YouTube | Real-world audio, various speakers |
| [Fake-or-Real](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) | TBD | Kaggle | TTS-generated fakes |

## Results

| Dataset | Accuracy | Recall (Fake) | F1 |
|---------|----------|---------------|-----|
| Original Eval | 99.7% | - | - |
| In-The-Wild | 42% | 12.4% | 17.6% |
| Fake-or-Real | TBD | TBD | TBD |

## Key Findings

**In-The-Wild results:**
- Model performs worse than random guessing (42% vs 50%)
- 87% of deepfakes pass undetected
- High confidence even on wrong predictions

## Project Structure

```
├── notebooks/
│   ├── 01_in_the_wild.ipynb    # In-The-Wild evaluation
│   ├── 02_fake_or_real.ipynb   # Fake-or-Real evaluation  
│   └── 03_comparison.ipynb     # Cross-dataset analysis
├── data/                       # Dataset files (not tracked)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/yourusername/deepfake-audio-eval
cd deepfake-audio-eval
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Model

[MelodyMachine/Deepfake-audio-detection-V2](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) - Wav2Vec2-based binary classifier fine-tuned for deepfake detection.

## Tech Stack

- HuggingFace Transformers
- librosa
- scikit-learn
- pandas, matplotlib, seaborn

## License

MIT

<div align="center">

# COME-448 Sentiment Analysis

### Comparing Transformer Models for Turkish Product Reviews

A reproducible NLP coursework project that fine-tunes and compares three transformer classifiers on Turkish product reviews, then reports their performance and training time.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-4.40%2B-yellow)](https://huggingface.co/docs/transformers)

[Run the script](#run-the-project) · [Models and data](#models-and-data) · [Outputs](#outputs)

</div>

---

## Project overview

This project studies binary sentiment classification for Turkish product reviews. It fine-tunes three pretrained transformer models on a balanced sample of the [Turkish Product Reviews dataset](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews) and compares their classification metrics and training speed.

## Models compared

| Model | Hugging Face checkpoint |
| --- | --- |
| BERTurk | `dbmdz/bert-base-turkish-cased` |
| DistilBERTurk | `dbmdz/distilbert-base-turkish-cased` |
| XLM-RoBERTa | `xlm-roberta-base` |

## Method

- Load Turkish product reviews from Hugging Face Datasets.
- Map positive and negative sentiment labels to a binary target.
- Build a balanced sample with **10,000 training** and **2,000 test** examples.
- Tokenize reviews separately for each model, with a maximum sequence length of 128.
- Fine-tune using the Hugging Face `Trainer`, with early stopping and the best validation F1 checkpoint.
- Compare accuracy, precision, recall, F1-score, and training time.
- Generate confusion matrices, metric comparisons, and ROC curves.

The script uses a fixed random seed (`42`) for sampling and training configuration. Reported metrics can still vary with hardware and library versions.

## Repository contents

| File | Purpose |
| --- | --- |
| `sentiment_analysis.py` | Complete training, evaluation, and visualization pipeline |
| `sentiment_analysis.ipynb` | Notebook version of the project |
| `sentiment_analysis_colab.ipynb` | Notebook prepared for Google Colab |
| `requirements.txt` | Python dependencies |

## Run the project

### Requirements

- Python 3
- Internet access to download the dataset and pretrained model checkpoints
- A CUDA-capable GPU is recommended; the script also supports CPU execution, which can take considerably longer

### 1. Clone the repository

```bash
git clone https://github.com/enesozturk0/COME-448-Sentiment-Analysis.git
cd COME-448-Sentiment-Analysis
```

### 2. Install dependencies

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Install the packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA setup by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 3. Train and evaluate the models

```bash
python sentiment_analysis.py
```

The first run downloads the dataset and model checkpoints. The script automatically selects CUDA when available and otherwise runs on CPU.

To explore the notebook version, launch Jupyter:

```bash
jupyter notebook sentiment_analysis.ipynb
```

The Colab notebook can also be opened directly in Google Colab after uploading or connecting the repository.

## Outputs

The script creates a `results/` directory next to `sentiment_analysis.py` and writes:

- `veri_seti_istatistikleri.png` — label counts and review-length visualizations
- `confusion_matrices.png` — confusion matrix for each successfully trained model
- `metrik_karsilastirma.png` — classification metrics and training-time comparison
- `roc_curves.png` — ROC curves with AUC values
- `sonuc_tablosu.csv` — summary metrics and training times

Model checkpoints are temporary and removed after each model run to reduce disk usage.

## Key dependencies

- [Transformers](https://huggingface.co/docs/transformers) and [Datasets](https://huggingface.co/docs/datasets) for pretrained models, tokenization, training, and data loading
- [PyTorch](https://pytorch.org/) for model execution
- [scikit-learn](https://scikit-learn.org/) for classification metrics and evaluation
- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/) for data handling and visualizations

## License

No license is currently specified in this repository.

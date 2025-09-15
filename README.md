# Lightbrain: Medical Small Language Model (LightBrain-Tiny-v1.0)


---

## Table of Contents
- [Project Overview](#project-overview)  
- [Goals and Objectives](#goals-and-objectives)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Directory Structure](#directory-structure)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Step 1: Preparing the Environment](#step-1-preparing-the-environment)  
  - [Step 2: Data Preparation](#step-2-data-preparation)  
  - [Step 3: Training the Model](#step-3-training-the-model)  
  - [Step 4: Evaluating the Model](#step-4-evaluating-the-model)  
  - [Step 5: Running Inference](#step-5-running-inference)  
- [Configuration Details](#configuration-details)  
- [Troubleshooting](#troubleshooting)  
- [Results and Examples](#results-and-examples)  
- [Limitations and Future Improvements](#limitations-and-future-improvements)  
- [Contributing](#contributing)  
- [License](#license)  
- [Credits and Acknowledgments](#credits-and-acknowledgments)  
- [Disclaimer](#disclaimer)  

---

## Project Overview
The **Lightbrain** project builds **LightBrain-Tiny-v1.0**, a small language model (SLM) from scratch using PyTorch, designed for **medical text generation**.  

- **Model size**: ~50-60 million parameters  
- **Domain**: PubMed QA or clinical notes  
- **Architecture**: GPT-like decoder-only transformer  
- **Purpose**: Educational & research exploration of medical domain language modeling  

> ⚠️ Not for clinical use. Research only.

### Key Features
- **From Scratch Implementation**: No external weights.  
- **Efficient Training**: Memory-mapped datasets + mixed-precision training.  
- **Medical Text Generation**: Generates summaries, abstracts, case descriptions.  
- **Modular Design**: Follows industry-standard project layout.  

---

## Goals and Objectives
- Develop a **lightweight SLM (~55M params)** for medical text generation.  
- Provide **end-to-end workflow** (data, tokenization, training, inference).  
- Ensure training feasibility on **consumer GPUs (e.g., Colab T4)**.  
- Achieve **reasonable perplexity/loss** on validation data.  
- Offer **educational insights** into transformer internals.  
- Promote **ethical, research-only AI development**.  

---

## Dataset
- **Source**: [BigBIO PubMed QA](https://huggingface.co/datasets/bigbio/pubmed_qa) or custom (PubMed abstracts, MIMIC-III).  
- **Description**: Biomedical/clinical question-answer pairs, abstracts, narratives.  
- **Stats (approximate)**:  
  - Train: ~180K entries  
  - Validation: ~20K entries  
  - Length: 200–500 tokens  

### Preprocessing
- Tokenized with **GPT-2 tokenizer (tiktoken)**.  
- Saved as **binary files**:  
  - `data/processed/train.bin`  
  - `data/processed/validation.bin`  

---

## Model Architecture
**LightBrain-Tiny-v1.0** is a scaled-down GPT-2 style **decoder-only transformer**:  

- **Vocab Size**: 50,257  
- **Embedding Dimension**: 384  
- **Layers**: 6  
- **Heads**: 6  
- **Context Window**: 128 tokens  
- **Dropout**: 0.1  
- **Params**: ~55M  

### Components
- Token + Positional embeddings  
- Causal Self-Attention (Flash Attention supported)  
- MLP with GELU activation  
- LayerNorm  
- Weight tying (embedding ↔ LM head)  

---

## 📂 Directory Structure

<details>
<summary><b>Click to expand</b></summary>

```bash
lightbrain-sml-from-scratch/
├── data/
│   ├── raw/pubmed_qa/          # Raw dataset
│   ├── processed/              # Tokenized data
│   └── external/pretrained/    # Optional pretrained weights
├── src/
│   ├── data/                   # Data prep
│   ├── models/                 # Model definition
│   ├── training/               # Training utilities
│   ├── inference/              # Generation scripts
│   └── visualization/          # Plots
├── notebooks/                  # Jupyter experiments
├── scripts/                    # Shell scripts (train, inference)
├── configs/                    # Config YAMLs
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── models/                     # Saved checkpoints
├── logs/                       # Logs & TensorBoard
├── requirements.txt
├── README.md
├── LICENSE
├── setup.py
└── environment.yml
```
</details>

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/lightbrain-sml-from-scratch.git
cd lightbrain-sml-from-scratch
pip install -r requirements.txt
```

Or, with conda:

```bash
conda env create -f environment.yml
conda activate lightbrain
```

---

## 📚 Requirements

```bash
datasets
tiktoken
torch
numpy
tqdm
matplotlib
pytest
```

---

## 🚀 Usage

### Training
```bash
python src/training/train.py --config configs/train_config.yaml
```

### Inference
```bash
python src/inference/generate.py --model models/checkpoint.pt --input "What is the treatment for pneumonia?"
```

### Visualization
```bash
python src/visualization/plot_losses.py
```

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgments

- PubMedQA dataset for biomedical Q&A
- Hugging Face `datasets` and `transformers`
- PyTorch deep learning framework

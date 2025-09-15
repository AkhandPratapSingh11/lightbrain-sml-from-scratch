Lightbrain: Medical Small Language Model (LightBrain-Tiny-v1.0)
(Note: Replace this placeholder with an actual banner image, e.g., a diagram of the model architecture with medical icons.)
Table of Contents

Project Overview
Goals and Objectives
Dataset
Model Architecture
Directory Structure
Requirements
Installation
Usage
Step 1: Preparing the Environment
Step 2: Data Preparation
Step 3: Training the Model
Step 4: Evaluating the Model
Step 5: Running Inference


Configuration Details
Troubleshooting
Results and Examples
Limitations and Future Improvements
Contributing
License
Credits and Acknowledgments
Disclaimer

Project Overview
The Lightbrain project builds LightBrain-Tiny-v1.0, a small language model (SLM) from scratch using PyTorch, designed for medical text generation. The model is trained on a medical dataset, such as PubMed QA or clinical notes, to handle medical terminology and generate coherent text for research purposes. LightBrain-Tiny-v1.0 is a transformer-based model inspired by GPT-like architectures, with approximately 50-60 million parameters for computational efficiency.
The implementation is provided in a Jupyter Notebook (notebooks/LightBrain_Tiny_v1.0_Scratch.ipynb) and modular Python scripts under src/, covering data preparation, tokenization, model definition, training, evaluation, and inference. This project is intended for educational and research purposes only, to explore language modeling in the medical domain, and is not suitable for clinical use.
Key features:

From Scratch Implementation: Built without external model weights, fully transparent.
Efficient Training: Uses memory-mapped files for large datasets and mixed-precision training for speed.
Medical Text Generation: Generates medical-related text, such as summaries or hypothetical case descriptions.
Modular Design: Organized for scalability and collaboration, following industry standards.

This README provides a comprehensive guide to replicating, running, and extending the Lightbrain project.
Goals and Objectives

Develop LightBrain-Tiny-v1.0, a lightweight SLM (~50-60M parameters) for medical text generation with accurate terminology.
Demonstrate an end-to-end workflow: data loading, tokenization, model architecture, training, and inference.
Enable training on consumer hardware (e.g., Google Colab with GPU).
Achieve reasonable perplexity/loss on medical validation data.
Provide insights into transformer components (causal self-attention, MLP blocks, positional embeddings) in a medical context.
Promote ethical AI development by emphasizing research-only use.

Dataset
The project uses a medical dataset, such as "bigbio/pubmed_qa" from Hugging Face or a custom collection of medical texts (e.g., PubMed abstracts, MIMIC-III clinical notes). Update the dataset loading code in src/data/prepare_data.py or the notebook to use your chosen dataset.

Description: A collection of medical texts (e.g., question-answer pairs, abstracts, or clinical narratives) focusing on biomedical and clinical language.
Size: Varies by dataset (e.g., PubMed QA has ~200K examples).
Preprocessing: Tokenized using GPT-2's tokenizer (tiktoken) and saved as binary files (data/processed/train.bin, data/processed/validation.bin) for efficient loading.
Why this dataset?: Provides domain-specific medical vocabulary and structures, enabling the model to learn medical text patterns without requiring massive general-domain data.

Dataset Statistics (approximate, based on PubMed QA):

Training examples: ~180K entries.
Validation examples: ~20K entries.
Average entry length: 200-500 tokens.

Note: Ensure compliance with dataset licenses and ethical guidelines (e.g., HIPAA for clinical data). Anonymize sensitive information before use.
Model Architecture
LightBrain-Tiny-v1.0 is a decoder-only transformer, scaled down from GPT-2, optimized for medical text:

Vocabulary Size: 50,257 (GPT-2 tokenizer; consider medical-specific tokenizers like BioBERT for better performance).
Embedding Dimension (n_embd): 384.
Number of Layers (n_layer): 6.
Number of Heads (n_head): 6.
Block Size (Context Window): 128 tokens.
Dropout: 0.1.
Total Parameters: ~55M (embeddings + positional + attention + MLP layers).
Components:
Token and Positional Embeddings.
Causal Self-Attention (with Flash Attention support for efficiency).
Feed-Forward MLP with GELU activation.
Layer Normalization.
Weight Tying (embedding and language model head).


Forward Pass: Computes logits for next-token prediction; uses cross-entropy loss.
Generation: Supports temperature scaling and top-k sampling for diverse outputs.

The model is defined in src/models/lightbrain_tiny.py, with helper classes like CausalSelfAttention, MLP, and Block.
Directory Structure
The project follows an industry-standard structure for modularity and scalability:
lightbrain-sml-from-scratch/
├── data/
│   ├── raw/pubmed_qa/          # Raw medical dataset
│   ├── processed/              # Tokenized data (train.bin, validation.bin)
│   └── external/pretrained/    # Optional pretrained weights
├── src/
│   ├── data/                   # Data loading and tokenization
│   ├── models/                 # LightBrain-Tiny-v1.0 architecture
│   ├── training/               # Training loop and utilities
│   ├── inference/              # Text generation scripts
│   └── visualization/          # Plotting utilities
├── notebooks/                  # Jupyter notebook for experimentation
├── scripts/                    # Shell scripts for tasks (train, inference)
├── configs/                    # Model and training configurations (YAML)
├── tests/                      # Unit tests
├── docs/                       # Documentation (architecture, API)
├── models/                     # Saved model weights
├── logs/                       # Training logs and TensorBoard
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── LICENSE                     # License file
├── .gitignore                  # Git ignore file
├── setup.py                    # Python package setup
└── environment.yml             # Conda environment file

Run create_project_structure.py to set up this structure (see Installation).
Requirements

Hardware:
CPU for basic runs; GPU (e.g., NVIDIA with CUDA) recommended for training.
Minimum: 16GB RAM, 8GB VRAM (e.g., Google Colab T4 GPU).


Software:
Python 3.10+.
Jupyter Notebook or Google Colab for interactive execution.


Libraries (listed in requirements.txt):
datasets: For loading medical datasets.
tiktoken: GPT-2 tokenizer (consider BioBERT tokenizer for medical terms).
torch: PyTorch for model and training.
numpy, tqdm: Utilities.
matplotlib: For plotting losses.
pytest: For unit testing.



Installation

Clone the Repository:
git clone https://github.com/your-username/lightbrain-sml-from-scratch.git
cd lightbrain-sml-from-scratch


Create Directory Structure:

Run the provided create_project_structure.py script to set up the project:python create_project_structure.py




Set Up Virtual Environment (recommended):
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate


Install Dependencies:Run the following in your terminal or use the provided script:
bash scripts/setup_env.sh

or
pip install -r requirements.txt


For GPU: Ensure CUDA-compatible PyTorch (e.g., pip install torch --index-url https://download.pytorch.org/whl/cu118 for CUDA 11.8).
For Conda: Use conda env create -f environment.yml.


Google Colab Setup (alternative for free GPU):

Upload notebooks/LightBrain_Tiny_v1.0_Scratch.ipynb to Colab.
Enable GPU: Runtime > Change runtime type > GPU.
Run !pip install commands in the notebook.



Usage
The project is implemented in notebooks/LightBrain_Tiny_v1.0_Scratch.ipynb for prototyping and modular scripts in src/ for production. Below are the steps to run the project.
Step 1: Preparing the Environment

Set up the directory structure and dependencies as described in Installation.
Ensure the medical dataset is placed in data/raw/pubmed_qa/ (e.g., download via datasets.load_dataset("bigbio/pubmed_qa")).

Step 2: Data Preparation

Run the data preparation script to tokenize the medical dataset and save binary files:python src/data/prepare_data.py


This generates data/processed/train.bin and data/processed/validation.bin. Expect 10-30 minutes for large datasets.
Update src/data/prepare_data.py with your dataset's specific loading logic (e.g., replace "roneneldan/TinyStories" with "bigbio/pubmed_qa").

Step 3: Training the Model

Run the training script:bash scripts/train_model.sh


Alternatively, use the notebook or src/training/train.py.
Duration: ~4-6 hours on Colab T4 GPU for 20,000 iterations.
Monitoring: Losses logged to logs/training.log and TensorBoard (logs/tensorboard/).
Output: Best model saved as models/lightbrain_tiny_v1.0.pt.
Hyperparameters (in configs/training_config.yaml):
Learning Rate: 1e-4 (with warmup and cosine decay).
Batch Size: 32.
Gradient Accumulation: 32 steps.
Mixed Precision: bfloat16/float16.



To resume training:
model.load_state_dict(torch.load('models/lightbrain_tiny_v1.0.pt'))

Step 4: Evaluating the Model

Evaluate using src/training/utils.py (contains estimate_loss):from src.training.utils import estimate_loss
losses = estimate_loss(model)
print(f"Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")


Expected Results: Validation loss ~2.5-3.5 (perplexity ~12-30), depending on dataset.

Step 5: Running Inference

Use the inference script or notebook:bash scripts/generate_text.sh


Example in Python:from src.inference.generate import generate
import tiktoken
enc = tiktoken.get_encoding("gpt2")
sentence = "The patient presented with symptoms of fever and cough."
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0).to(device)
output = generate(model, context, max_new_tokens=200, temperature=1.0, top_k=50)
print(enc.decode(output.squeeze().tolist()))


Outputs medical-like text (see Results).

Configuration Details

Model Config (configs/model_config.yaml):
vocab_size: 50257, block_size: 128, n_layer: 6, n_head: 6, n_embd: 384, dropout: 0.1, bias: true.


Training Config (configs/training_config.yaml):
Optimizer: AdamW with weight decay (0.1).
Scheduler: Linear warmup (1,000 steps) + Cosine decay to min_lr (5e-4).
Gradient Clipping: Max norm 0.5.
Mixed Precision: bfloat16/float16.


Device Handling: Auto-detects CUDA; uses AMP for efficiency.

Edit configs for experimentation (e.g., increase n_layer or adjust learning_rate).
Troubleshooting

Memory Errors: Reduce batch_size or block_size in configs/training_config.yaml. Increase gradient_accumulation_steps.
Dependency Issues: Update datasets (pip install -U datasets) or check PyTorch CUDA compatibility (torch.cuda.is_available()).
Slow Tokenization: Use multi-core processing (num_proc=8) in src/data/prepare_data.py.
Inference Inaccuracies: Train longer, use a medical-specific tokenizer, or lower temperature for coherence.
File Not Found: Verify data/processed/train.bin and data/processed/validation.bin exist.
Medical Data Issues: Ensure dataset format matches expected structure (e.g., text field); truncate long sequences.

For persistent issues, check PyTorch forums or open an issue in the lightbrain-sml-from-scratch repository.
Results and Examples

Training Curve: Losses decrease steadily, visualized in logs/tensorboard/ or notebook.
Inference Examples:
Prompt: "The patient presented with symptoms of fever and cough."
Output: Hypothetical continuation, e.g., describing diagnoses or tests (simplistic due to model size).


Prompt: "In a study on cardiovascular disease,"
Output: Abstract-like text on medical topics.




Performance: Generates 200-token passages in <1 second on GPU. Outputs are coherent but may lack clinical accuracy.

See notebooks/LightBrain_Tiny_v1.0_Scratch.ipynb for full examples.
Limitations and Future Improvements

Limitations:
Small model size limits complexity; outputs may include factual errors or hallucinations.
Trained on public medical data; not validated for clinical use.
GPT-2 tokenizer may split medical terms poorly; consider BioBERT tokenizer.
Potential dataset biases; no handling for rare medical terms.


Improvements:
Scale up model (e.g., more layers/heads).
Use medical-specific tokenizers (e.g., BioBERT, ClinicalBERT).
Implement PEFT (e.g., LoRA) for efficient fine-tuning.
Integrate with Hugging Face Transformers for deployment.
Use larger datasets (e.g., Med-PaLM, MIMIC-IV).
Add evaluation metrics (e.g., ROUGE, domain-specific accuracy).



Contributing
Contributions are welcome! Fork the lightbrain-sml-from-scratch repo, make changes, and submit a pull request. Focus areas:

Optimizing data processing or training.
Adding medical-specific tokenizers or datasets.
Enhancing inference for better coherence.
Improving tests in tests/.

License
MIT License. See LICENSE for details.
Credits and Acknowledgments

Authors: Lightbrain Team.
Inspirations:
Andrej Karpathy's nanoGPT (data prep and training utilities).
PyTorch documentation.
Hugging Face Datasets and TikToken.


Dataset: Depending on choice (e.g., BigBIO for PubMed QA).
Thanks to open-source communities for tools and tutorials.

For questions, contact [your-email@example.com]. Last updated: September 15, 2025.
Disclaimer
LightBrain-Tiny-v1.0 is for educational and research purposes only. Generated text is not medical advice and may contain errors. Do not use for diagnosing, treating, or making medical decisions. Consult qualified healthcare professionals for clinical applications. Ensure compliance with data privacy laws (e.g., HIPAA) when using medical datasets.

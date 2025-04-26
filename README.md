# Fine-Tuning Recipes

This repository contains recipes and examples for fine-tuning various types of models using HuggingFace libraries. The focus is on creating high-quality models for different use cases, with special attention to:

- Embedding models for semantic search and RAG applications
- Large Language Models (LLMs) for specific tasks
- Other model types (coming soon)

## Structure

```
.
├── embedding-models/         # Fine-tuning recipes for embedding models
│   ├── datasets/            # Dataset creation and processing scripts
│   ├── models/              # Model training and evaluation scripts
│   └── notebooks/           # Jupyter notebooks for experimentation
│
├── llm-models/              # Fine-tuning recipes for LLMs
│   ├── datasets/            # Dataset creation and processing scripts
│   ├── models/              # Model training and evaluation scripts
│   └── notebooks/           # Jupyter notebooks for experimentation
│
├── requirements/            # Python dependencies for different model types
│   ├── embedding.txt       # Dependencies for embedding models
│   └── llm.txt            # Dependencies for LLM fine-tuning
│
└── README.md               # This file
```

## Getting Started

1. Choose the type of model you want to fine-tune
2. Install the appropriate dependencies:
```bash
# For embedding models
pip install -r requirements/embedding.txt

# For LLM fine-tuning
pip install -r requirements/llm.txt
```

3. Follow the specific instructions in each model type's directory

## Features

### Embedding Models
- Dataset creation for embedding model fine-tuning
- Support for various loss functions (TripletLoss, MultipleNegativesRankingLoss, etc.)
- Evaluation using NDCG and other metrics
- Support for multiple embedding models (E5, BGE, etc.)

#### Usage Examples

1. Create training and test datasets:
```bash
# Create training dataset
python embedding-models/datasets/create_synthetic_dataset.py \
    --num_samples 1000 \
    --output_path embedding-models/datasets/embedding_synthetic_training_dataset.csv

# Create test dataset
python embedding-models/datasets/create_synthetic_dataset.py \
    --num_samples 200 \
    --output_path embedding-models/datasets/embedding_synthetic_test_dataset.csv
```

2. Train an embedding model:
```bash
python embedding-models/models/train.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --train_dataset_path embedding-models/datasets/embedding_synthetic_training_dataset.csv \
    --output_dir models/all-MiniLM-L6-v2 \
    --batch_size 32 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --evaluation_steps 100 \
    --use_wandb \
    --wandb_project embedding-fine-tuning
```

3. Evaluate the trained model on the test set:
```bash
python embedding-models/models/evaluate.py \
    --model models/all-MiniLM-L6-v2 \
    --test_dataset_path embedding-models/datasets/embedding_synthetic_test_dataset.csv \
    --batch_size 32
```

Complete workflow example:
```bash
# 1. Create datasets
python embedding-models/datasets/create_synthetic_dataset.py --num_samples 1000 --output_path embedding-models/datasets/embedding_synthetic_training_dataset.csv
python embedding-models/datasets/create_synthetic_dataset.py --num_samples 200 --output_path embedding-models/datasets/embedding_synthetic_test_dataset.csv

# 2. Train the model
python embedding-models/models/train.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --train_dataset_path embedding-models/datasets/embedding_synthetic_training_dataset.csv \
    --output_dir models/all-MiniLM-L6-v2 \
    --batch_size 32 \
    --epochs 3 \
    --use_wandb

# 3. Evaluate on test set
python embedding-models/models/evaluate.py \
    --model models/all-MiniLM-L6-v2 \
    --test_dataset_path embedding-models/datasets/embedding_synthetic_test_dataset.csv \
    --batch_size 32
```

The scripts will:
1. Create synthetic datasets for both training and testing
2. Train a model using the training dataset with the specified parameters
3. Evaluate the model's performance on the test set using NDCG and other metrics

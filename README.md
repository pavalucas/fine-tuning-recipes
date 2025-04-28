# Fine-Tuning Recipes

This repository contains recipes and examples for fine-tuning embedding models using HuggingFace libraries. The focus is on creating high-quality models for semantic search and RAG applications.

## Structure

```
.
├── embedding-models/         # Fine-tuning recipes for embedding models
│   ├── datasets/            # Dataset creation and processing scripts
│   ├── models/              # Model training and evaluation scripts
│   └── notebooks/           # Jupyter notebooks for experimentation
│
├── requirements/            # Python dependencies
│   └── embedding.txt       # Dependencies for embedding models
│
└── README.md               # This file
```

## Getting Started

1. Install the required dependencies:
```bash
pip install -r requirements/embedding.txt
```

2. Follow the instructions below to fine-tune your embedding model

## Features

### Embedding Models
- Dataset creation for embedding model fine-tuning
- Support for both local CSV files and Hugging Face datasets
- Evaluation using NDCG and other metrics
- Support for multiple embedding models (E5, BGE, etc.)
- Flexible column name configuration for different dataset formats

### Dataset Formats

The scripts support two types of datasets:

1. **Triplet Format**: Contains three columns in order: anchor, positive, negative
   - Default column names: `anchor,positive,negative`
   - Example:
     ```csv
     anchor,positive,negative
     "What is Python?","Python is a programming language...","Java is a programming language..."
     ```

2. **Pair Format**: Contains two columns in order: anchor, positive
   - Default column names: `anchor,positive`
   - Example:
     ```csv
     anchor,positive
     "What is Python?","Python is a programming language..."
     ```

You can specify custom column names using the `--triplet_columns` or `--pair_columns` arguments.

#### Usage Examples

1. Using a local CSV file with default column names:
```bash
# First, create the synthetic dataset
python embedding-models/datasets/create_synthetic_dataset.py \
    --num_samples 1000 \
    --output_path embedding-models/datasets/embedding_synthetic_training_dataset.csv

# Then train the model using the created dataset
python embedding-models/models/train.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --dataset_source embedding-models/datasets/embedding_synthetic_training_dataset.csv \
    --output_dir models/all-MiniLM-L6-v2 \
    --batch_size 32 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --evaluation_steps 100 \
    --use_wandb \
    --wandb_project embedding-fine-tuning
```

2. Using a local CSV file with custom column names:
```bash
# For triplets with custom column names
python embedding-models/models/train.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --dataset_source embedding-models/datasets/embedding_synthetic_training_dataset.csv \
    --output_dir models/all-MiniLM-L6-v2 \
    --triplet_columns "question,relevant_answer,irrelevant_answer" \
    --use_wandb

# For pairs with custom column names
python embedding-models/models/train.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --dataset_source embedding-models/datasets/embedding_synthetic_training_dataset.csv \
    --output_dir models/all-MiniLM-L6-v2 \
    --pair_columns "text1,text2" \
    --use_wandb
```

3. Using a Hugging Face dataset:
```bash
python embedding-models/models/train.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --dataset_source sentence-transformers/quora-duplicates:triplet \
    --output_dir models/all-MiniLM-L6-v2 \
    --batch_size 32 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --evaluation_steps 100 \
    --use_wandb \
    --wandb_project embedding-fine-tuning
```

Note: For Hugging Face datasets that require a configuration (like Quora duplicates), specify the config using the format `dataset_name:config_name`. For example:
- `sentence-transformers/quora-duplicates:triplet` for triplet format
- `sentence-transformers/quora-duplicates:pair` for pair format
- `sentence-transformers/quora-duplicates:pair-class` for pair classification format

4. Evaluate the trained model on the test set:
```bash
# Using a local CSV file with default column names
python embedding-models/models/evaluate.py \
    --model models/all-MiniLM-L6-v2 \
    --dataset_source embedding-models/datasets/embedding_synthetic_test_dataset.csv \
    --batch_size 32

# Using a local CSV file with custom column names
python embedding-models/models/evaluate.py \
    --model models/all-MiniLM-L6-v2 \
    --dataset_source embedding-models/datasets/embedding_synthetic_test_dataset.csv \
    --triplet_columns "question,relevant_answer,irrelevant_answer" \
    --batch_size 32

# Using a Hugging Face dataset
python embedding-models/models/evaluate.py \
    --model models/all-MiniLM-L6-v2 \
    --dataset_source sentence-transformers/quora-duplicates:triplet \
    --batch_size 32
```

Complete workflow example with Hugging Face dataset:
```bash
# 1. Train the model using Quora duplicates dataset
python embedding-models/models/train.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --dataset_source sentence-transformers/quora-duplicates:triplet \
    --output_dir models/all-MiniLM-L6-v2 \
    --batch_size 32 \
    --epochs 3 \
    --use_wandb

# 2. Evaluate on test set
python embedding-models/models/evaluate.py \
    --model models/all-MiniLM-L6-v2 \
    --dataset_source sentence-transformers/quora-duplicates:triplet \
    --batch_size 32
```

The scripts will:
1. Load the dataset from either a local CSV file or Hugging Face (automatically detected)
2. Train a model using the training dataset with the specified parameters
3. Evaluate the model's performance on the test set using NDCG and other metrics

## Weights & Biases Integration

The training script integrates with Weights & Biases for experiment tracking. To use it:

1. Install Weights & Biases:
   ```bash
   pip install wandb
   ```

2. Login to Weights & Biases:
   ```bash
   wandb login
   ```

3. Run the training script with `--use_wandb` flag:
   ```bash
   python embedding-models/models/train.py --use_wandb ...
   ```

The script will log:
- Training configuration
- Training loss
- Evaluation metrics
- Model checkpoints
import os
from pathlib import Path
from typing import Optional, Union, Dict, List
import logging
import wandb
import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation
)
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import ndcg_score
import numpy as np
from sentence_transformers import util
from sklearn.metrics import accuracy_score
import argparse
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_wandb(project_name: str, config: Dict) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize Weights & Biases with proper error handling."""
    try:
        # Check if wandb is logged in
        if not wandb.api.api_key:
            logging.error("Weights & Biases is not logged in. Please run 'wandb login' first.")
            return None
        
        # Try to initialize wandb
        run = wandb.init(
            project=project_name,
            config=config
        )
        logging.info("Successfully initialized Weights & Biases")
        return run
    except wandb.errors.UsageError as e:
        logging.error(f"Weights & Biases usage error: {str(e)}")
        logging.error("Please make sure you're logged in and have the correct permissions.")
        return None
    except Exception as e:
        logging.error(f"Failed to initialize Weights & Biases: {str(e)}")
        return None

def load_dataset_from_source(
    dataset_source: str,
    split: str = "train",
    triplet_columns: str = "anchor,positive,negative",
    pair_columns: str = "anchor,positive"
) -> List[InputExample]:
    """Load dataset from either a local CSV file or Hugging Face dataset.
    
    Args:
        dataset_source: Either a path to a local CSV file or a Hugging Face dataset name
                       For Hugging Face datasets that require a config, use format: dataset_name:config_name
        split: Dataset split to use (train/test/validation)
        triplet_columns: Comma-separated column names for triplet format in order: anchor,positive,negative
                        Default: "anchor,positive,negative"
        pair_columns: Comma-separated column names for pair format in order: anchor,positive
                     Default: "anchor,positive"
    
    Returns:
        List of InputExample objects
    """
    # Split column names
    triplet_cols = [col.strip() for col in triplet_columns.split(',')]
    pair_cols = [col.strip() for col in pair_columns.split(',')]
    
    logging.info(f"Loading dataset from source: {dataset_source}")
    
    # Check if the source is a local file
    if os.path.exists(dataset_source):
        logging.info("Loading from local CSV file")
        df = pd.read_csv(dataset_source)
        examples = []
        
        # Check if we have all required columns for triplet format
        if all(col in df.columns for col in triplet_cols):
            for _, row in df.iterrows():
                examples.append(InputExample(
                    texts=[
                        row[triplet_cols[0]],  # anchor
                        row[triplet_cols[1]],  # positive
                        row[triplet_cols[2]]   # negative
                    ],
                    label=1.0
                ))
        # Check if we have all required columns for pair format
        elif all(col in df.columns for col in pair_cols):
            for _, row in df.iterrows():
                examples.append(InputExample(
                    texts=[
                        row[pair_cols[0]],  # anchor
                        row[pair_cols[1]]   # positive
                    ],
                    label=1.0
                ))
        else:
            raise ValueError(
                f"Dataset must contain either triplet columns ({triplet_cols}) "
                f"or pair columns ({pair_cols})"
            )
    else:
        # Try to load from Hugging Face
        logging.info("Loading from Hugging Face dataset")
        try:
            # Check if dataset source includes a config
            if ':' in dataset_source:
                dataset_name, config_name = dataset_source.split(':')
                dataset = load_dataset(dataset_name, config_name, split=split)
            else:
                dataset = load_dataset(dataset_source, split=split)
            
            examples = []
            # Check if we have all required columns for triplet format
            if all(col in dataset.features for col in triplet_cols):
                for item in dataset:
                    examples.append(InputExample(
                        texts=[
                            item[triplet_cols[0]],  # anchor
                            item[triplet_cols[1]],  # positive
                            item[triplet_cols[2]]   # negative
                        ],
                        label=1.0
                    ))
            # Check if we have all required columns for pair format
            elif all(col in dataset.features for col in pair_cols):
                for item in dataset:
                    examples.append(InputExample(
                        texts=[
                            item[pair_cols[0]],  # anchor
                            item[pair_cols[1]]   # positive
                        ],
                        label=1.0
                    ))
            else:
                raise ValueError(
                    f"Dataset must contain either triplet columns ({triplet_cols}) "
                    f"or pair columns ({pair_cols})"
                )
        except Exception as e:
            if 'Config name is missing' in str(e):
                available_configs = str(e).split('available configs: ')[1].split('\n')[0]
                raise ValueError(
                    f"Dataset {dataset_source} requires a configuration. "
                    f"Available configs: {available_configs}. "
                    f"Please specify the config using format: dataset_name:config_name"
                )
            raise ValueError(f"Failed to load dataset from {dataset_source}. Error: {str(e)}")
    
    logging.info(f"Loaded {len(examples)} examples")
    return examples

def evaluate_model(model, examples, batch_size=16):
    """Evaluate the model on a set of examples."""
    logging.info("Starting model evaluation...")
    
    # Compute embeddings for all texts
    texts = []
    for example in examples:
        texts.extend([example.texts[0], example.texts[1], example.texts[2]])
    
    logging.info(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    # Calculate similarities
    logging.info("Calculating similarities...")
    similarities = []
    y_true = []
    
    for i in range(0, len(embeddings), 3):
        query_emb = embeddings[i]
        pos_emb = embeddings[i + 1]
        neg_emb = embeddings[i + 2]
        
        # Calculate cosine similarities
        pos_sim = util.cos_sim(query_emb, pos_emb).item()
        neg_sim = util.cos_sim(query_emb, neg_emb).item()
        
        # Store similarities and ground truth
        similarities.append([pos_sim, neg_sim])
        y_true.append([1, 0])  # Positive document should be ranked first
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(similarities)
    
    # Calculate metrics
    logging.info("Computing metrics...")
    accuracy = accuracy_score(y_true.argmax(axis=1), y_score.argmax(axis=1))
    ndcg = ndcg_score(y_true, y_score)
    
    # Calculate average similarities
    avg_pos_sim = np.mean([sim[0] for sim in similarities])
    avg_neg_sim = np.mean([sim[1] for sim in similarities])
    
    metrics = {
        'accuracy': accuracy,
        'ndcg': ndcg,
        'avg_pos_sim': avg_pos_sim,
        'avg_neg_sim': avg_neg_sim
    }
    
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics

def train_embedding_model(
    model_name: str,
    dataset_source: str,
    output_dir: str,
    batch_size: int = 32,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    evaluation_steps: int = 100,
    use_wandb: bool = True,
    device: Optional[str] = None,
    wandb_project: str = "embedding-fine-tuning",
    triplet_columns: str = "anchor,positive,negative",
    pair_columns: str = "anchor,positive"
) -> SentenceTransformer:
    """
    Train an embedding model using SentenceTransformers.
    
    Args:
        model_name: Name of the base model from HuggingFace
        dataset_source: Either a path to a local CSV file or a Hugging Face dataset name
        output_dir: Directory to save the trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        warmup_steps: Number of warmup steps
        evaluation_steps: Steps between evaluations
        use_wandb: Whether to use Weights & Biases for logging
        device: Device to use for training (cuda/cpu)
        wandb_project: Weights & Biases project name
        triplet_columns: Comma-separated column names for triplet format in order: anchor,positive,negative
        pair_columns: Comma-separated column names for pair format in order: anchor,positive
    """
    logging.info(f"Starting training with model: {model_name}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
    
    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        config = {
            "model_name": model_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "dataset_source": dataset_source,
            "triplet_columns": triplet_columns,
            "pair_columns": pair_columns
        }
        wandb_run = initialize_wandb(wandb_project, config)
        if wandb_run is None:
            logging.warning("Continuing without Weights & Biases logging")
            use_wandb = False
    
    # Load the dataset
    train_examples = load_dataset_from_source(
        dataset_source,
        split="train",
        triplet_columns=triplet_columns,
        pair_columns=pair_columns
    )
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(train_examples))
    train_data = train_examples[:train_size]
    val_data = train_examples[train_size:]
    logging.info(f"Split dataset into {len(train_data)} training and {len(val_data)} validation examples")
    
    # Initialize the model
    logging.info(f"Initializing model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Define loss function (MultipleNegativesRankingLoss for triplets)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Create evaluator
    evaluator = evaluation.TripletEvaluator(
        anchors=[ex.texts[0] for ex in val_data],
        positives=[ex.texts[1] for ex in val_data],
        negatives=[ex.texts[2] for ex in val_data],
        name='validation'
    )
    
    # Train the model
    logging.info("Starting training...")
    
    # Define callback for logging training loss
    def log_callback(score, epoch, steps, loss):
        if use_wandb and wandb_run:
            wandb.log({
                "eval_score": score,
                "train_loss": loss,
                "epoch": epoch,
                "steps": steps
            })
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        callback=log_callback
    )
    
    # Final evaluation
    metrics = evaluate_model(model, val_data, batch_size)
    
    if use_wandb and wandb_run is not None:
        wandb.log(metrics)
        wandb.finish()
    
    return model

def compare_models(
    baseline_model: SentenceTransformer,
    fine_tuned_model: SentenceTransformer,
    val_data: List[InputExample],
    batch_size: int = 16
) -> Dict[str, Dict[str, float]]:
    """Compare baseline and fine-tuned models on validation set."""
    logging.info("Starting model comparison...")
    
    # Evaluate both models
    baseline_metrics = evaluate_model(baseline_model, val_data, batch_size)
    fine_tuned_metrics = evaluate_model(fine_tuned_model, val_data, batch_size)
    
    # Calculate relative improvements
    improvements = {
        'accuracy': (fine_tuned_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100,
        'ndcg': (fine_tuned_metrics['ndcg'] - baseline_metrics['ndcg']) / baseline_metrics['ndcg'] * 100,
        'avg_pos_sim': (fine_tuned_metrics['avg_pos_sim'] - baseline_metrics['avg_pos_sim']) / baseline_metrics['avg_pos_sim'] * 100,
        'avg_neg_sim': (fine_tuned_metrics['avg_neg_sim'] - baseline_metrics['avg_neg_sim']) / baseline_metrics['avg_neg_sim'] * 100
    }
    
    comparison = {
        'baseline': baseline_metrics,
        'fine_tuned': fine_tuned_metrics,
        'improvements': improvements
    }
    
    # Log comparison results
    logging.info("\nModel Comparison Results:")
    logging.info("Baseline Model:")
    for metric, value in baseline_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
    
    logging.info("\nFine-tuned Model:")
    for metric, value in fine_tuned_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
    
    logging.info("\nRelative Improvements (%):")
    for metric, value in improvements.items():
        logging.info(f"  {metric}: {value:.2f}%")
    
    return comparison

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train an embedding model')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                      help='Name of the base model from HuggingFace (default: sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('--dataset_source', type=str, required=True,
                      help='Either a path to a local CSV file or a Hugging Face dataset name')
    parser.add_argument('--output_dir', type=str, default='models/all-MiniLM-L6-v2',
                      help='Directory to save the trained model (default: models/all-MiniLM-L6-v2)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs (default: 3)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate for training (default: 2e-5)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                      help='Number of warmup steps (default: 100)')
    parser.add_argument('--evaluation_steps', type=int, default=100,
                      help='Steps between evaluations (default: 100)')
    parser.add_argument('--use_wandb', action='store_true',
                      help='Use Weights & Biases for logging')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for training (cuda/cpu)')
    parser.add_argument('--wandb_project', type=str, default='embedding-fine-tuning',
                      help='Weights & Biases project name (default: embedding-fine-tuning)')
    parser.add_argument('--triplet_columns', type=str, default="anchor,positive,negative",
                      help='Comma-separated column names for triplet format in order: anchor,positive,negative (default: anchor,positive,negative)')
    parser.add_argument('--pair_columns', type=str, default="anchor,positive",
                      help='Comma-separated column names for pair format in order: anchor,positive (default: anchor,positive)')
    args = parser.parse_args()
    
    # Train the model
    model = train_embedding_model(
        model_name=args.model,
        dataset_source=args.dataset_source,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.evaluation_steps,
        use_wandb=args.use_wandb,
        device=args.device,
        wandb_project=args.wandb_project,
        triplet_columns=args.triplet_columns,
        pair_columns=args.pair_columns
    )

if __name__ == "__main__":
    main() 
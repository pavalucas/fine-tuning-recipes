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

def load_dataset(csv_path: Union[str, Path]) -> List[InputExample]:
    """Load the dataset from CSV and convert to InputExamples."""
    logging.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Dataset loaded with {len(df)} rows")
    
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(
            texts=[row['query'], row['positive_doc'], row['negative_doc']],
            label=1.0  # Positive example
        ))
    
    logging.info(f"Converted {len(examples)} examples to InputExamples")
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
    train_dataset_path: Union[str, Path],
    output_dir: Union[str, Path],
    batch_size: int = 32,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    evaluation_steps: int = 100,
    use_wandb: bool = True,
    device: Optional[str] = None,
    wandb_project: str = "embedding-fine-tuning"
) -> SentenceTransformer:
    """
    Train an embedding model using SentenceTransformers.
    
    Args:
        model_name: Name of the base model from HuggingFace
        train_dataset_path: Path to the training dataset CSV
        output_dir: Directory to save the trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        warmup_steps: Number of warmup steps
        evaluation_steps: Steps between evaluations
        use_wandb: Whether to use Weights & Biases for logging
        device: Device to use for training (cuda/cpu)
        wandb_project: Weights & Biases project name
        
    Returns:
        SentenceTransformer: Trained model
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
        }
        wandb_run = initialize_wandb(wandb_project, config)
        if wandb_run is None:
            logging.warning("Continuing without Weights & Biases logging")
            use_wandb = False
    
    # Load the dataset
    train_examples = load_dataset(train_dataset_path)
    
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
    
    # Define loss function (TripletLoss for triplets)
    train_loss = losses.TripletLoss(model=model)
    
    # Create evaluator
    evaluator = evaluation.TripletEvaluator(
        anchors=[ex.texts[0] for ex in val_data],
        positives=[ex.texts[1] for ex in val_data],
        negatives=[ex.texts[2] for ex in val_data],
        name='validation'
    )
    
    # Train the model
    logging.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        callback=lambda score, epoch, steps: wandb.log({"eval_score": score}) if use_wandb and wandb_run else None
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
    parser.add_argument('--train_dataset_path', type=str, default='embedding-models/datasets/embedding_training_dataset_100.csv',
                      help='Path to the training dataset CSV (default: embedding-models/datasets/embedding_training_dataset_100.csv)')
    parser.add_argument('--output_dir', type=str, default='models/all-MiniLM-L6-v2',
                      help='Directory to save the trained model (default: models/all-MiniLM-L6-v2)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size (default: 16)')
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
    args = parser.parse_args()
    
    # Load validation data
    val_data = load_dataset(args.train_dataset_path)
    train_size = int(0.8 * len(val_data))
    val_data = val_data[train_size:]
    
    # Initialize baseline model
    logging.info(f"Initializing baseline model: {args.model}")
    baseline_model = SentenceTransformer(args.model)
    
    # Train fine-tuned model
    logging.info("Training fine-tuned model...")
    fine_tuned_model = train_embedding_model(
        model_name=args.model,
        train_dataset_path=args.train_dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.evaluation_steps,
        use_wandb=args.use_wandb,
        device=args.device,
        wandb_project=args.wandb_project
    )
    
    # Compare models
    comparison = compare_models(baseline_model, fine_tuned_model, val_data)
    
    # Log to wandb if available
    if wandb.run is not None:
        wandb.log({
            'baseline_accuracy': comparison['baseline']['accuracy'],
            'baseline_ndcg': comparison['baseline']['ndcg'],
            'baseline_avg_pos_sim': comparison['baseline']['avg_pos_sim'],
            'baseline_avg_neg_sim': comparison['baseline']['avg_neg_sim'],
            'improvement_accuracy': comparison['improvements']['accuracy'],
            'improvement_ndcg': comparison['improvements']['ndcg'],
            'improvement_avg_pos_sim': comparison['improvements']['avg_pos_sim'],
            'improvement_avg_neg_sim': comparison['improvements']['avg_neg_sim']
        })

if __name__ == "__main__":
    main() 
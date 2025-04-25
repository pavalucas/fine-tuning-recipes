import os
from pathlib import Path
from typing import Optional, Union, Dict, List
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

def load_dataset(csv_path: Union[str, Path]) -> List[InputExample]:
    """Load the dataset from CSV and convert to InputExamples."""
    df = pd.read_csv(csv_path)
    examples = []
    
    for _, row in df.iterrows():
        examples.append(InputExample(
            texts=[row['query'], row['positive_doc'], row['negative_doc']],
            label=1.0  # Positive example
        ))
    
    return examples

def evaluate_model(
    model: SentenceTransformer,
    test_dataset: List[InputExample],
    batch_size: int = 32
) -> Dict[str, float]:
    """Evaluate the model using NDCG and other metrics."""
    # Extract queries and documents
    queries = [example.texts[0] for example in test_dataset]
    positive_docs = [example.texts[1] for example in test_dataset]
    negative_docs = [example.texts[2] for example in test_dataset]
    
    # Compute embeddings
    query_embeddings = model.encode(queries, batch_size=batch_size)
    positive_embeddings = model.encode(positive_docs, batch_size=batch_size)
    negative_embeddings = model.encode(negative_docs, batch_size=batch_size)
    
    # Calculate similarities
    positive_similarities = np.sum(query_embeddings * positive_embeddings, axis=1)
    negative_similarities = np.sum(query_embeddings * negative_embeddings, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(positive_similarities > negative_similarities)
    
    # Calculate NDCG
    y_true = np.array([[1, 0]])  # Ideal ranking: positive doc first
    y_score = np.array([positive_similarities, negative_similarities]).T
    ndcg = ndcg_score(y_true, y_score)
    
    return {
        'accuracy': accuracy,
        'ndcg': ndcg,
        'avg_positive_similarity': np.mean(positive_similarities),
        'avg_negative_similarity': np.mean(negative_similarities)
    }

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
    device: Optional[str] = None
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
        
    Returns:
        SentenceTransformer: Trained model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="embedding-fine-tuning",
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
            }
        )
    
    # Load the dataset
    train_examples = load_dataset(train_dataset_path)
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(train_examples))
    train_data = train_examples[:train_size]
    val_data = train_examples[train_size:]
    
    # Initialize the model
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
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        use_wandb=use_wandb
    )
    
    # Final evaluation
    metrics = evaluate_model(model, val_data, batch_size)
    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    if use_wandb:
        wandb.log(metrics)
        wandb.finish()
    
    return model

if __name__ == "__main__":
    # Example usage
    model = train_embedding_model(
        model_name="intfloat/multilingual-e5-large",  # Good for multilingual tasks
        train_dataset_path="embedding_training_dataset.csv",
        output_dir="models/finetuned_e5",
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
        warmup_steps=100,
        evaluation_steps=100,
        use_wandb=True
    ) 
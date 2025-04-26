import os
from pathlib import Path
from typing import Optional, Union, List
from sentence_transformers import SentenceTransformer, evaluation
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import ndcg_score
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def evaluate_model(
    model: Union[str, SentenceTransformer],
    test_dataset_path: Union[str, Path],
    batch_size: int = 32,
    device: Optional[str] = None
) -> dict:
    """
    Evaluate an embedding model using NDCG and other metrics.
    
    Args:
        model: Either a model name or a SentenceTransformer instance
        test_dataset_path: Path to the test dataset CSV file
        batch_size: Evaluation batch size
        device: Device to use for evaluation (cuda/cpu)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
    
    # Load or initialize the model
    if isinstance(model, str):
        logging.info(f"Loading model: {model}")
        model = SentenceTransformer(model, device=device)
    
    # Load the test dataset
    logging.info(f"Loading test dataset from {test_dataset_path}")
    test_df = pd.read_csv(test_dataset_path)
    logging.info(f"Loaded {len(test_df)} test examples")
    
    # Prepare evaluation data
    queries = test_df["query"].tolist()
    positive_docs = test_df["positive_doc"].tolist()
    negative_docs = test_df["negative_doc"].tolist()
    
    # Compute embeddings
    logging.info("Computing embeddings...")
    query_embeddings = model.encode(queries, batch_size=batch_size, show_progress_bar=True)
    pos_embeddings = model.encode(positive_docs, batch_size=batch_size, show_progress_bar=True)
    neg_embeddings = model.encode(negative_docs, batch_size=batch_size, show_progress_bar=True)
    
    # Calculate similarities
    logging.info("Calculating similarities...")
    pos_similarities = []
    neg_similarities = []
    
    for i in range(len(queries)):
        pos_sim = np.dot(query_embeddings[i], pos_embeddings[i])
        neg_sim = np.dot(query_embeddings[i], neg_embeddings[i])
        pos_similarities.append(pos_sim)
        neg_similarities.append(neg_sim)
    
    # Calculate metrics
    logging.info("Computing metrics...")
    accuracy = np.mean([1 if pos > neg else 0 for pos, neg in zip(pos_similarities, neg_similarities)])
    avg_pos_sim = np.mean(pos_similarities)
    avg_neg_sim = np.mean(neg_similarities)
    
    # Calculate NDCG
    y_true = np.array([[1, 0] for _ in range(len(queries))])
    y_score = np.array([[pos, neg] for pos, neg in zip(pos_similarities, neg_similarities)])
    ndcg = ndcg_score(y_true, y_score)
    
    metrics = {
        'accuracy': accuracy,
        'ndcg': ndcg,
        'avg_pos_sim': avg_pos_sim,
        'avg_neg_sim': avg_neg_sim
    }
    
    logging.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate an embedding model')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                      help='Name of the model from HuggingFace (default: sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('--test_dataset_path', type=str, default='embedding-models/datasets/embedding_synthetic_test_dataset.csv',
                      help='Path to the test dataset CSV (default: embedding-models/datasets/embedding_synthetic_test_dataset.csv)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Evaluation batch size (default: 32)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for evaluation (cuda/cpu)')
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_model(
        model=args.model,
        test_dataset_path=args.test_dataset_path,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("\nFinal Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 
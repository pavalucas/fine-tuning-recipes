import os
from pathlib import Path
from typing import Optional, Union, List
from sentence_transformers import SentenceTransformer, evaluation
from datasets import load_from_disk
import torch
import numpy as np
from sklearn.metrics import ndcg_score

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
        test_dataset_path: Path to the test dataset
        batch_size: Evaluation batch size
        device: Device to use for evaluation (cuda/cpu)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load or initialize the model
    if isinstance(model, str):
        model = SentenceTransformer(model, device=device)
    
    # Load the test dataset
    test_dataset = load_from_disk(test_dataset_path)
    
    # Prepare evaluation data
    queries = test_dataset["query"]
    documents = test_dataset["positive_doc"]
    
    # Compute embeddings
    query_embeddings = model.encode(queries, batch_size=batch_size)
    doc_embeddings = model.encode(documents, batch_size=batch_size)
    
    # Compute similarity scores
    similarity_scores = np.dot(query_embeddings, doc_embeddings.T)
    
    # Compute NDCG
    # For each query, the relevant document is at position 0
    true_relevance = np.zeros_like(similarity_scores)
    np.fill_diagonal(true_relevance, 1)
    
    ndcg_scores = []
    for i in range(len(queries)):
        ndcg_scores.append(ndcg_score(
            [true_relevance[i]],
            [similarity_scores[i]],
            k=10
        ))
    
    # Compute mean NDCG
    mean_ndcg = np.mean(ndcg_scores)
    
    # Create evaluator for additional metrics
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=documents,
        relevant_docs={i: [i] for i in range(len(queries))},
        show_progress_bar=True
    )
    
    # Run evaluation
    metrics = evaluator(model)
    metrics["ndcg@10"] = mean_ndcg
    
    return metrics

if __name__ == "__main__":
    # Example usage
    metrics = evaluate_model(
        model="intfloat/multilingual-e5-large",
        test_dataset_path="datasets/qa_dataset",
        batch_size=32
    )
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 
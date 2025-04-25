import json
from pathlib import Path
from typing import List, Dict, Union
from datasets import Dataset
import pandas as pd

def create_qa_dataset(
    queries: List[str],
    positive_docs: List[str],
    negative_docs: List[str] = None,
    output_path: Union[str, Path] = None
) -> Dataset:
    """
    Create a QA dataset for fine-tuning embedding models.
    
    Args:
        queries: List of query strings
        positive_docs: List of positive document strings
        negative_docs: Optional list of negative document strings
        output_path: Optional path to save the dataset
        
    Returns:
        Dataset: HuggingFace dataset object
    """
    if len(queries) != len(positive_docs):
        raise ValueError("Number of queries must match number of positive documents")
        
    if negative_docs and len(queries) != len(negative_docs):
        raise ValueError("Number of queries must match number of negative documents")
    
    data = {
        "query": queries,
        "positive_doc": positive_docs,
    }
    
    if negative_docs:
        data["negative_doc"] = negative_docs
    
    dataset = Dataset.from_dict(data)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
    
    return dataset

def create_symmetric_dataset(
    texts: List[str],
    pairs: List[Dict[str, int]],
    scores: List[float] = None,
    output_path: Union[str, Path] = None
) -> Dataset:
    """
    Create a symmetric dataset for semantic similarity tasks.
    
    Args:
        texts: List of text strings
        pairs: List of dictionaries with 'text1_idx' and 'text2_idx' keys
        scores: Optional list of similarity scores
        output_path: Optional path to save the dataset
        
    Returns:
        Dataset: HuggingFace dataset object
    """
    data = {
        "text1": [texts[pair["text1_idx"]] for pair in pairs],
        "text2": [texts[pair["text2_idx"]] for pair in pairs],
    }
    
    if scores:
        data["score"] = scores
    
    dataset = Dataset.from_dict(data)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
    
    return dataset

if __name__ == "__main__":
    # Example usage
    queries = ["What is machine learning?", "How does deep learning work?"]
    positive_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data.",
        "Deep learning is a type of machine learning that uses artificial neural networks with multiple layers."
    ]
    negative_docs = [
        "The weather is nice today.",
        "I like to eat pizza."
    ]
    
    # Create QA dataset
    qa_dataset = create_qa_dataset(
        queries=queries,
        positive_docs=positive_docs,
        negative_docs=negative_docs,
        output_path="datasets/qa_dataset"
    )
    
    # Create symmetric dataset
    texts = ["Hello world", "Hi there", "Greetings"]
    pairs = [
        {"text1_idx": 0, "text2_idx": 1},
        {"text1_idx": 1, "text2_idx": 2}
    ]
    scores = [0.8, 0.6]
    
    sym_dataset = create_symmetric_dataset(
        texts=texts,
        pairs=pairs,
        scores=scores,
        output_path="datasets/symmetric_dataset"
    ) 
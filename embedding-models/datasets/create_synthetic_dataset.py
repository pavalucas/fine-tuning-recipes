import json
from pathlib import Path
from typing import List, Dict, Union
from datasets import Dataset
import pandas as pd
import random
import argparse
import logging

def generate_synthetic_query(topic: str) -> str:
    """Generate a synthetic query about a topic."""
    query_templates = [
        "What is the ruling on {topic}?",
        "What are the details of {topic}?",
        "How is {topic} handled?",
        "What is the procedure for {topic}?",
        "What are the requirements for {topic}?"
    ]
    return random.choice(query_templates).format(topic=topic)

def generate_synthetic_document(topic: str, case_id: str) -> str:
    """Generate a synthetic legal document."""
    document_templates = [
        "In the case number {case_id}, the court ruled on {topic}. The judgment was based on the following considerations: {details}",
        "The court's decision in case {case_id} regarding {topic} established that {details}",
        "Case {case_id} addressed the matter of {topic}. The ruling stated that {details}"
    ]
    
    details = [
        "the evidence presented was sufficient to establish the claim",
        "the requirements for the claim were met",
        "the legal conditions were satisfied",
        "the claim was supported by relevant documentation",
        "the applicable regulations were properly followed"
    ]
    
    template = random.choice(document_templates)
    detail = random.choice(details)
    return template.format(case_id=case_id, topic=topic, details=detail)

def create_synthetic_dataset(
    num_samples: int = 100,
    output_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Create a synthetic dataset for embedding model training.
    
    Args:
        num_samples: Number of samples to generate
        output_path: Optional path to save the dataset
        
    Returns:
        DataFrame: Dataset with query, positive_doc, negative_doc, and case IDs
    """
    # List of legal topics
    topics = [
        "contract disputes",
        "property rights",
        "commercial transactions",
        "employment law",
        "intellectual property",
        "bankruptcy proceedings",
        "taxation matters",
        "corporate governance",
        "consumer protection",
        "environmental regulations"
    ]
    
    # Generate samples
    samples = []
    for i in range(num_samples):
        # Select a topic for the positive case
        topic = random.choice(topics)
        
        # Generate case IDs
        positive_case_id = f"case_{i+1}"
        negative_case_id = f"case_{i+1000}"  # Different range to avoid overlap
        
        # Generate query and documents
        query = generate_synthetic_query(topic)
        positive_doc = generate_synthetic_document(topic, positive_case_id)
        
        # Select a differsent topic for the negative document
        negative_topic = random.choice([t for t in topics if t != topic])
        negative_doc = generate_synthetic_document(negative_topic, negative_case_id)
        
        samples.append({
            "query": query,
            "positive_doc": positive_doc,
            "negative_doc": negative_doc,
            "positive_case_id": positive_case_id,
            "negative_case_id": negative_case_id
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    # Save to CSV if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    return df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create synthetic dataset for embedding model training')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of samples to generate (default: 100)')
    parser.add_argument('--output_path', type=str, default='embedding-models/datasets/embedding_syntetic_training_dataset.csv',
                      help='Path to save the output dataset (default: embedding-models/datasets/embedding_syntetic_training_dataset.csv)')
    args = parser.parse_args()
    
    # Create synthetic dataset
    logging.info(f"Creating synthetic dataset with {args.num_samples} samples")
    dataset = create_synthetic_dataset(
        num_samples=args.num_samples,
        output_path=args.output_path
    )
    
    # Print sample of the dataset
    print("\nSample of the created dataset:")
    print(dataset.head())
    logging.info(f"Dataset saved to {args.output_path}")

if __name__ == "__main__":
    main() 
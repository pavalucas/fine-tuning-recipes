import os
import logging
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, evaluation
from datasets import load_dataset
from typing import List, Optional, Union
from pathlib import Path
import argparse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_dataset_from_source(
    dataset_source: str,
    split: str = "test",
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

def evaluate_model(
    model: Union[str, Path],
    dataset_source: str,
    batch_size: int = 32,
    device: Optional[str] = None,
    triplet_columns: str = "anchor,positive,negative",
    pair_columns: str = "anchor,positive"
) -> dict:
    """
    Evaluate a trained embedding model.
    
    Args:
        model: Path to the trained model or model name
        dataset_source: Either a path to a local CSV file or a Hugging Face dataset name
        batch_size: Evaluation batch size
        device: Device to use for evaluation (cuda/cpu)
        triplet_columns: Comma-separated column names for triplet format in order: anchor,positive,negative
                        Default: "anchor,positive,negative"
        pair_columns: Comma-separated column names for pair format in order: anchor,positive
                     Default: "anchor,positive"
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
    
    # Load the model
    logging.info(f"Loading model from {model}")
    model = SentenceTransformer(str(model), device=device)
    
    # Load the test dataset
    test_examples = load_dataset_from_source(
        dataset_source,
        split="test",
        triplet_columns=triplet_columns,
        pair_columns=pair_columns
    )
    
    # Create evaluator
    evaluator = evaluation.TripletEvaluator(
        anchors=[ex.texts[0] for ex in test_examples],
        positives=[ex.texts[1] for ex in test_examples],
        negatives=[ex.texts[2] for ex in test_examples],
        name='test'
    )
    
    # Run evaluation
    logging.info("Starting evaluation...")
    metrics = evaluator(model)
    
    logging.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate an embedding model')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the trained model or model name')
    parser.add_argument('--dataset_source', type=str, required=True,
                      help='Either a path to a local CSV file or a Hugging Face dataset name')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Evaluation batch size (default: 32)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for evaluation (cuda/cpu)')
    parser.add_argument('--triplet_columns', type=str, default="anchor,positive,negative",
                      help='Comma-separated column names for triplet format in order: anchor,positive,negative (default: anchor,positive,negative)')
    parser.add_argument('--pair_columns', type=str, default="anchor,positive",
                      help='Comma-separated column names for pair format in order: anchor,positive (default: anchor,positive)')
    args = parser.parse_args()
    
    metrics = evaluate_model(
        model=args.model,
        dataset_source=args.dataset_source,
        batch_size=args.batch_size,
        device=args.device,
        triplet_columns=args.triplet_columns,
        pair_columns=args.pair_columns
    )
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 
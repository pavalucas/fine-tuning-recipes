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

### LLM Models
- Dataset preparation for instruction fine-tuning
- Support for different fine-tuning approaches (LoRA, QLoRA, etc.)
- Evaluation metrics for LLM performance
- Support for various LLM architectures

## References

### Embedding Models
- [Fine-tune Embedding Model for RAG](https://www.philschmid.de/fine-tune-embedding-model-for-rag)
- [Enhancing Arabic Text Understanding in RAG Models](https://medium.com/@alroumi.abdulmajeed/enhancing-arabic-text-understanding-in-rag-models-through-fine-tuning-embeddings-bede568d66aa)
- [GATE: A Challenge Set for Gender-Ambiguous Translation Examples](https://arxiv.org/abs/2402.05672)

### LLM Models
- [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/docs/peft/index)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

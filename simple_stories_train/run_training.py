#!/usr/bin/env python3
"""
SimpleStories Training Pipeline

This script provides a complete end-to-end pipeline for preprocessing
and training a language model on the SimpleStories or TinyStories dataset.

Key differences from the main training repo: 
    Dataloader is changed to take in the json file as input
    33M model config is added to model_configs

Usage:
    python train_simple_stories.py --dataset=DATASET --output_dir=OUTPUT_DIR --topk_tokens=topk_tokens

Options:
    dataset: Choose 'tinystories' or 'simplestories' [default: simplestories]
    output_dir: Directory to save processed data [default: "processed_data"]
    topk_tokens: Maximum number of tokens to keep [default: 10000]
    preprocess: Whether to run preprocessing before training [default: False]
    batch_size: Batch size for training [default: 16]
    num_iterations: Number of training iterations [default: 60000]
    learning_rate: Learning rate for training [default: 1e-4]

TODO:
    Make the script default to huggingface format if JSON is not given or preprocessing is not added
"""

import json
from collections import Counter
from pathlib import Path

import datasets
import fire
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm
from train_llama import Config, DatasetConfig, main
from transformers import GPT2TokenizerFast


class Pipeline:
    """SimpleStories Training Pipeline"""
    
    def __init__(self, 
                 dataset="simplestories",
                 output_dir="processed_data",
                 topk_tokens=10000,
                 preprocess=False,
                 batch_size=4,
                 num_iterations=60000,
                 learning_rate=1e-4):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            dataset: Which dataset to use ('tinystories' or 'simplestories')
            output_dir: Directory to save processed data
            topk_tokens: Maximum number of tokens to keep in vocabulary
            preprocess: Whether to run preprocessing before training
            batch_size: Batch size for training
            num_iterations: Number of training iterations
            learning_rate: Learning rate for training
        """
        # Validate dataset choice
        if dataset not in ["tinystories", "simplestories"]:
            raise ValueError(f"Dataset must be 'tinystories' or 'simplestories', got '{dataset}'")
            
        self.dataset = dataset
        self.output_dir = output_dir
        self.topk_tokens = topk_tokens
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.run()
    
    def run(self):
        """Run the complete pipeline: optional preprocessing and mandatory training"""
        
        print(f"Starting SimpleStories training pipeline with {self.dataset} dataset")
        print(f"Output directory: {self.output_dir}")
        print(f"Max tokens: {self.topk_tokens}")
        
        # Step 1: Preprocess the dataset if requested
        output_dir = Path(self.output_dir)
        train_file = output_dir / "train.json"
        val_file = output_dir / "validation.json"
        
        if self.preprocess:
            print("\nStarting preprocessing step...")
            if self.dataset == "tinystories":
                data_files = preprocess_tinystories(self.output_dir, self.topk_tokens)
            else:  # simplestories
                data_files = preprocess_simplestories(self.output_dir, self.topk_tokens)
        else:
            ## TODO: Replace it with training from huggingface directly
            # Check if processed files exist
            if not (train_file.exists() and val_file.exists()):
                raise FileNotFoundError(
                    "Processed data files not found. Please run with --preprocess=True first "
                    "or ensure the processed files exist in the output directory."
                )
            data_files = {
                'train': str(train_file),
                'validation': str(val_file)
            }
            print("\nUsing existing preprocessed data files...")
        
        # Step 2: Train the model (mandatory)
        print("\nPreparing training configuration...")
        config = prepare_training(
            data_files,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations,
            learning_rate=self.learning_rate
        )
        
        print("\nStarting model training...")
        main(config)
        
        print("\nPipeline completed successfully!")
        return "Success"


def preprocess_tinystories(output_dir: str | Path, topk_tokens: int) -> dict[str, str]:
    """
    Preprocess TinyStories dataset:
    1. Find top N tokens
    2. For each story, remove text corresponding to tokens not in top N
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading TinyStories dataset...")
    dataset: DatasetDict = load_dataset("roneneldan/TinyStories")  # type: ignore
    
    # Initialize tokenizer
    print("Initializing GPT2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125M')
    
    # Count token frequencies
    print("Counting token frequencies...")
    token_counts = Counter()
    
    for split in ['train', 'validation']:
        for item in tqdm(dataset[split], desc=f"Processing {split}"):
            tokens = tokenizer.encode(item['text']) # type: ignore
            token_counts.update(tokens)
    
    # Get set of top tokens
    top_tokens = set(token for token, _ in token_counts.most_common(topk_tokens))
    
    # Process stories - remove text corresponding to non-top tokens
    print("Processing stories...")
    splits = {}
    
    for split in ['train', 'validation']:
        processed_stories = []
        
        for item in tqdm(dataset[split], desc=f"Processing {split}"):
            text = item['text'] # type: ignore
            encoding = tokenizer(text, return_offsets_mapping=True)
            tokens = encoding['input_ids']
            token_offsets = encoding['offset_mapping']
            
            # Keep track of which parts of text to keep
            filtered_text = ""
            
            for token, (start, end) in zip(tokens, token_offsets):
                if token in top_tokens:
                    filtered_text += text[start:end]
            
            processed_stories.append({'text': filtered_text})
        
        # Save processed stories
        output_file = output_dir / f"{split}.json"
        with open(output_file, "w") as f:
            json.dump(processed_stories, f)
        
        splits[split] = str(output_file)
        print(f"Saved {len(processed_stories)} stories for {split}")
    
    return splits


def preprocess_simplestories(output_dir: str | Path, topk_tokens: int) -> dict[str, str]:
    """
    Preprocess SimpleStories dataset:
    1. Find top N tokens
    2. For each story, remove text corresponding to tokens not in top N
    3. Save in JSON format compatible with training script
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading SimpleStories dataset...")
    dataset = datasets.load_dataset("lennart-finke/SimpleStories")
    
    # Initialize tokenizer
    print("Initializing GPT2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125M')
    
    # Count token frequencies
    print("Counting token frequencies...")
    token_counts = Counter()
    
    for split in ['train', 'test']:
        for item in tqdm(dataset[split], desc=f"Processing {split}"): # type: ignore
            tokens = tokenizer.encode(item['story']) # type: ignore
            token_counts.update(tokens)
    
    # Get set of top tokens
    top_tokens = set(token for token, _ in token_counts.most_common(topk_tokens))
    
    # Process stories - remove text corresponding to non-top tokens
    print("Processing stories...")
    splits = {}
    train_split = 'train'
    val_split = 'test'  # SimpleStories uses 'test' instead of 'validation'
    
    # Process and save train data
    processed_stories = []
    for item in tqdm(dataset[train_split], desc=f"Processing {train_split}"): # type: ignore
        text = item['story'] # type: ignore
        encoding = tokenizer(text, return_offsets_mapping=True)
        tokens = encoding['input_ids']
        token_offsets = encoding['offset_mapping']
        
        filtered_text = ""
        for token, (start, end) in zip(tokens, token_offsets):
            if token in top_tokens:
                filtered_text += text[start:end]
        
        processed_stories.append({'text': filtered_text})
    
    # Save processed train stories in the format expected by the training script
    train_file = output_dir / "train.json"
    with open(train_file, "w") as f:
        json.dump(processed_stories, f)
    splits['train'] = str(train_file)
    print(f"Saved {len(processed_stories)} stories for train")
    
    # Process and save validation data
    processed_stories = []
    for item in tqdm(dataset[val_split], desc=f"Processing {val_split}"): # type: ignore
        text = item['story'] # type: ignore
        encoding = tokenizer(text, return_offsets_mapping=True)
        tokens = encoding['input_ids']
        token_offsets = encoding['offset_mapping']
        
        filtered_text = ""
        for token, (start, end) in zip(tokens, token_offsets):
            if token in top_tokens:
                filtered_text += text[start:end]
        
        processed_stories.append({'text': filtered_text})
    
    # Save processed validation stories
    val_file = output_dir / "validation.json"
    with open(val_file, "w") as f:
        json.dump(processed_stories, f)
    splits['validation'] = str(val_file)
    print(f"Saved {len(processed_stories)} stories for validation")
    
    return splits


def prepare_training(data_files: dict[str, str], batch_size=16, num_iterations=60000, learning_rate=1e-4):
    """Prepare the training configuration using the processed data files"""
    # Initialize tokenizer with special tokens for our story format
    tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125M')
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Save tokenizer to a local path
    tokenizer_path = Path("gpt2_tokenizer")
    tokenizer_path.mkdir(exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    
    # Create training config
    config = Config(
        # Model and data configuration
        model_name="d2",
        train_dataset_config=DatasetConfig(
            name="json",
            is_tokenized=False,
            tokenizer_file_path=str(tokenizer_path / "tokenizer.json"),
            streaming=True,
            n_ctx=512,
            seed=42,
            column_name="text",  # Adjust this to match your JSON structure
            data_files=data_files['train']
        ),
        val_dataset_config=DatasetConfig(
            name="json",
            is_tokenized=False,
            tokenizer_file_path=str(tokenizer_path / "tokenizer.json"),
            streaming=True,
            n_ctx=512,
            seed=42,
            column_name="text",  # Adjust this to match your JSON structure
            data_files=data_files['validation']
        ),
        
        # Training hyperparameters
        batch_size=batch_size,
        total_batch_size=batch_size * 1024,  # Adjust based on batch_size
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        warmup_iters=2000,
        weight_decay=0.1,
        grad_clip=1.0,
        
        val_loss_every=50,
        val_max_steps=20,
        sample_every=100,
        inference_only=False,
        learning_rate_decay_frac=1.0,
        tensorcores=True,
        
        # Hardware and optimization
        device="cuda",
        compile=True,
        flash_attention=True,
        dtype="bfloat16",
        zero_stage=0,
        
        # Output configuration
        output_dir=Path("training_outputs"),
        wandb_project=None,  # Set to None if you don't want to use wandb
        intermediate_checkpoints=True,

        # tokenizer option
        eos_token_id="<|endoftext|>"
    )
    
    return config

if __name__ == "__main__":
    fire.Fire(Pipeline)
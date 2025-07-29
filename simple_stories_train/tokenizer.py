"""
This file is inspired from Nix Goldowsky-Dill's adaption of the tokenizer in https://github.com/juand-r/tiny_tokenizer.
"""

from itertools import chain
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, Replace
from tokenizers.normalizers import Sequence as NormSequence
from tokenizers.pre_tokenizers import Digits, Punctuation, Sequence, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tqdm import tqdm

from simple_stories_train.dataloaders import DatasetConfig, create_data_loader

OUT_DIR = Path("tokenizer")

# Define common affixes for special handling based on morphological analysis of the dataset
COMMON_PREFIXES = ["un", "re"]
COMMON_SUFFIXES = ["ed", "ing", "ly", "er", "ness"]


def clean_dataset(dataset="SimpleStories/SimpleStories") -> list[str]:
    """
    Load and clean the dataset, implementing lowercase strategy.
    Returns the entire cleaned dataset as a list of stories.
    """
    print(f"Loading and cleaning dataset: {dataset}")
    dataset = load_dataset(dataset, trust_remote_code=False)
    trans = str.maketrans(
        {"\u201d": '"', "\u201c": '"', "\u2019": "'", "\u2018": "'", "\u2014": "-", "\u2026": "..."}
    )

    cleaned_data = [
        s.translate(trans).encode("ascii", "ignore").decode("ascii").lower()
        for s in dataset["train"]["story"]  # pyright: ignore
    ]

    print(f"Cleaned {len(cleaned_data)} stories")
    return cleaned_data


def create_validation_split(dataset="SimpleStories/SimpleStories") -> DatasetDict:
    """
    Create train/validation splits for model training (not tokenizer training).
    This is separate from tokenizer training and only used for analysis.
    """
    cleaned = clean_dataset(dataset)

    # Split into train and validation sets
    n_train = int(len(cleaned) * 0.9)
    train, validation = cleaned[:n_train], cleaned[n_train:]

    train_ds = Dataset.from_dict(dict(story=train))
    validation_ds = Dataset.from_dict(dict(story=validation))

    return DatasetDict({"train": train_ds, "validation": validation_ds})


def create_tokenizer(vocab_size=4096) -> Tokenizer:
    """
    Create a tokenizer with integrated affix handling using Split pre-tokenizers.

    Args:
        vocab_size: The target vocabulary size for the tokenizer

    Returns:
        A configured Tokenizer object ready for training
    """
    print(f"Creating tokenizer with target vocabulary size: {vocab_size}")

    # Initialize WordPiece tokenizer
    tokenizer = Tokenizer(
        WordPiece(
            unk_token="[UNK]"  # type: ignore
        )
    )

    # Set normalizers (lowercase everything)
    tokenizer.normalizer = NormSequence(
        [
            Lowercase(),
            Replace("``", '"'),
            Replace("''", '"'),
        ]  # type: ignore
    )

    # Set up the pre-tokenizer sequence
    tokenizer.pre_tokenizer = Sequence(
        [Whitespace(), Punctuation(), Digits(individual_digits=True)]
    )  # type: ignore

    # Add post-processor for special tokens
    tokenizer.post_processor = TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 1)])  # type: ignore

    tokenizer.decoder = WordPieceDecoder(prefix="##")  # type: ignore

    return tokenizer


def train_tokenizer(data: list[str], vocab_size: int = 4096) -> Tokenizer:
    """
    Train the tokenizer with the specified vocabulary size and cleaned data.

    Args:
        data: List of cleaned text strings to train on
        vocab_size: The target vocabulary size

    Returns:
        Trained Tokenizer object
    """
    print(f"Training tokenizer on {len(data)} stories with vocab_size {vocab_size}...")

    tokenizer = create_tokenizer(vocab_size)

    special_tokens = ["[UNK]", "[EOS]"]
    affixes = COMMON_PREFIXES + COMMON_SUFFIXES

    # Train the tokenizer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=special_tokens, initial_alphabet=affixes
    )

    tokenizer.train_from_iterator(data, trainer=trainer, length=len(data))
    print("Tokenizer training completed")

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, tokenizer_name: str) -> str:
    """
    Save tokenizer to file.

    Args:
        tokenizer: The tokenizer to save
        tokenizer_name: The filename for the tokenizer

    Returns:
        The full path where tokenizer was saved
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = f"{OUT_DIR}/{tokenizer_name}"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer_path


def prune_tokenizer(tokenizer: Tokenizer, dataset_texts: list[str]) -> Tokenizer:
    """
    Prune tokenizer by removing unused tokens and reordering IDs sequentially.

    Args:
        tokenizer: Trained tokenizer object
        dataset_texts: List of text strings to check token usage against

    Returns:
        Pruned Tokenizer object with sequential IDs
    """
    original_vocab_size = len(tokenizer.get_vocab())
    print(f"Original vocabulary size: {original_vocab_size}")

    # Always keep special tokens (IDs 0 and 1)
    special_tokens = {0, 1}

    # Find used tokens in dataset
    used_token_ids = set()
    for text in tqdm(dataset_texts, desc="Tokenizing dataset"):
        encoded = tokenizer.encode(text)
        used_token_ids.update(encoded.ids)

    # Keep both used and special tokens
    all_needed_tokens = used_token_ids | special_tokens

    print(f"Used tokens: {len(all_needed_tokens)}")
    print(f"Removing: {original_vocab_size - len(all_needed_tokens)} tokens")

    if len(all_needed_tokens) == original_vocab_size:
        print("No tokens to remove!")
        return tokenizer

    # Create new vocabulary with sequential IDs
    new_vocab = {}
    for new_id, old_id in enumerate(sorted(all_needed_tokens)):
        token_text = tokenizer.id_to_token(old_id)
        new_vocab[token_text] = new_id

    print(f"New vocabulary size: {len(new_vocab)}")

    # Create new tokenizer
    new_tokenizer: Tokenizer = Tokenizer(WordPiece(vocab=new_vocab, unk_token="[UNK]"))  # type: ignore

    # Add special tokens back
    new_tokenizer.add_special_tokens(
        [AddedToken("[UNK]", special=True), AddedToken("[EOS]", special=True)]
    )

    # Copy settings from original
    new_tokenizer.normalizer = tokenizer.normalizer  # type: ignore
    new_tokenizer.pre_tokenizer = tokenizer.pre_tokenizer  # type: ignore
    new_tokenizer.post_processor = tokenizer.post_processor  # type: ignore
    new_tokenizer.decoder = tokenizer.decoder  # type: ignore

    return new_tokenizer


def test_tokenizer(filepath: str, dataset: str = "SimpleStories/SimpleStories") -> None:
    """
    Test the trained tokenizer on sample data.
    """
    dataset_name = dataset
    split = "train"

    context_width = 512
    dataset_config = DatasetConfig(
        name=dataset_name,
        is_tokenized=False,
        tokenizer_file_path=filepath,
        streaming=True,
        split=split,
        n_ctx=context_width,
        seed=42,
        column_name="story",
    )

    batch_size = 1
    buffer_size = 1000
    global_seed = 0

    loader, tokenizer = create_data_loader(
        dataset_config, batch_size, buffer_size, global_seed, ddp_rank=0, ddp_world_size=1
    )
    batch = next(iter(loader))
    words = tokenizer.decode_batch(batch["input_ids"].tolist(), skip_special_tokens=False)
    print("Sample tokenization:")
    print(words)


def load_tokenizer(tokenizer_name="simplestories-4096.json") -> Tokenizer:
    """
    Load a tokenizer from file.

    Args:
        tokenizer_name: The filename of the tokenizer to load
    """
    return Tokenizer.from_file(f"{OUT_DIR}/{tokenizer_name}")


def print_split_words(story_tokens: list[str]) -> None:
    for i, token in enumerate(story_tokens):
        if token.startswith("##") and (i == 0 or not story_tokens[i - 1].startswith("##")):
            word_start = story_tokens[i - 1] if i > 0 else ""
            word_parts = [token]

            for next_token in story_tokens[i + 1 :]:
                if not next_token.startswith("##"):
                    break
                word_parts.append(next_token)

            print(f"{word_start} {' '.join(word_parts)}")


def analysis(tokenizer_name="simplestories-4096.json") -> None:
    """
    Analyze tokenizer performance using a validation split.
    This creates a fresh validation split for analysis purposes only.

    Args:
        tokenizer_name: The filename of the tokenizer to analyze
    """
    tokenizer = load_tokenizer(tokenizer_name)
    # Create validation split for analysis
    validation_stories = create_validation_split(dataset="SimpleStories/SimpleStories")[
        "validation"
    ]["story"]
    tokenized_stories = [tokenizer.encode(story).tokens for story in validation_stories]

    all_tokens = list(chain.from_iterable(tokenized_stories))
    partial_word_toks = len([token for token in all_tokens if token.startswith("##")])

    print(f"Tokens per story: {len(all_tokens) / len(validation_stories):.2f}")
    print(f"Proportion of partial word tokens: {partial_word_toks / len(all_tokens):.2%}")

    for story_idx, story_tokens in enumerate(tokenized_stories[:25]):
        print(f"\nStory {story_idx}")
        print_split_words(story_tokens)


if __name__ == "__main__":
    vocab_size = 4096
    dataset_name = "SimpleStories/SimpleStories"

    cleaned_data = clean_dataset(dataset=dataset_name)

    tokenizer = train_tokenizer(data=cleaned_data, vocab_size=vocab_size)

    pruned_tokenizer = prune_tokenizer(tokenizer, cleaned_data)

    save_tokenizer(pruned_tokenizer, "simplestories-tokenizer.json")

"""Simple test for tokenizer pruning functionality."""

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from simple_stories_train.tokenizer import prune_tokenizer


# Create tokenizer once for all tests
def setup_tokenizer():
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))  # type: ignore
    tokenizer.normalizer = Lowercase()  # type: ignore
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore

    # High vocab size with minimal data ensures many unused tokens
    trainer = WordPieceTrainer(vocab_size=200, special_tokens=["[UNK]", "[EOS]"])
    tokenizer.train_from_iterator(["hello world", "hello there", "world peace"], trainer=trainer)
    return tokenizer


# Global tokenizer and test data for reuse
TEST_TOKENIZER = setup_tokenizer()
TEST_DATA = ["hello world", "hello there", "world peace"]


def test_special_tokens_preserved():
    test_data = ["hello world"]
    pruned = prune_tokenizer(TEST_TOKENIZER, test_data)
    vocab = pruned.get_vocab()

    assert "[UNK]" in vocab and "[EOS]" in vocab
    assert vocab["[UNK]"] in [0, 1] and vocab["[EOS]"] in [0, 1]


def test_unused_tokens_removed():
    original_size = len(TEST_TOKENIZER.get_vocab())
    pruned = prune_tokenizer(TEST_TOKENIZER, ["hello"])  # Very limited data

    assert len(pruned.get_vocab()) < original_size


def test_functionality_preserved():
    pruned = prune_tokenizer(TEST_TOKENIZER, ["hello world"])
    encoded = pruned.encode("hello world")
    decoded = pruned.decode(encoded.ids)

    assert decoded == "hello world"


def test_sequential_ids():
    pruned = prune_tokenizer(TEST_TOKENIZER, ["hello"])
    token_ids = sorted(pruned.get_vocab().values())

    assert token_ids == list(range(len(token_ids)))

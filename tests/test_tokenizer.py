"""Simple test for tokenizer pruning functionality."""

from collections.abc import Generator

from simple_stories_train.tokenizer import prune_tokenizer, train_tokenizer


def create_test_data_iterator(test_data: list[str]) -> Generator[str, None, None]:
    """Create iterator from test data."""
    yield from test_data


def create_test_tokenizer():
    """Create a fresh tokenizer for testing."""
    train_data = ["hello world", "hello there", "world peace", "simple stories"]
    train_iter = create_test_data_iterator(train_data)
    return train_tokenizer(train_iter, vocab_size=200)


def test_special_tokens_preserved():
    """Verify special tokens exist in both original and pruned tokenizers."""
    tokenizer = create_test_tokenizer()

    vocab_orig = tokenizer.get_vocab()
    assert "[UNK]" in vocab_orig and "[EOS]" in vocab_orig

    test_data = ["hello world"]
    data_iter = create_test_data_iterator(test_data)
    pruned = prune_tokenizer(data_iter, tokenizer)
    vocab_pruned = pruned.get_vocab()
    assert "[UNK]" in vocab_pruned and "[EOS]" in vocab_pruned
    assert vocab_pruned["[UNK]"] in [0, 1] and vocab_pruned["[EOS]"] in [0, 1]


def test_unused_tokens_removed():
    """Verify pruning removes unused tokens from vocabulary."""
    tokenizer = create_test_tokenizer()

    original_size = len(tokenizer.get_vocab())
    test_data = ["hello"]
    data_iter = create_test_data_iterator(test_data)

    pruned = prune_tokenizer(data_iter, tokenizer)

    assert len(pruned.get_vocab()) < original_size


def test_functionality_preserved():
    """Verify encode/decode functionality works before and after pruning."""
    tokenizer = create_test_tokenizer()

    encoded_orig = tokenizer.encode("hello world")
    decoded_orig = tokenizer.decode(encoded_orig.ids)
    assert decoded_orig == "hello world"

    test_data = ["hello world"]
    data_iter = create_test_data_iterator(test_data)
    pruned = prune_tokenizer(data_iter, tokenizer)
    encoded_pruned = pruned.encode("hello world")
    decoded_pruned = pruned.decode(encoded_pruned.ids)
    assert decoded_pruned == "hello world"


def test_sequential_ids():
    """Verify pruned tokenizer has sequential token IDs starting from 0."""
    tokenizer = create_test_tokenizer()

    token_ids_orig = sorted(tokenizer.get_vocab().values())
    assert token_ids_orig[0] >= 0
    assert len(token_ids_orig) == len(set(token_ids_orig))

    test_data = ["hello"]
    data_iter = create_test_data_iterator(test_data)
    pruned = prune_tokenizer(data_iter, tokenizer)
    token_ids_pruned = sorted(pruned.get_vocab().values())
    assert token_ids_pruned == list(range(len(token_ids_pruned)))


def test_eos_appended():
    """Verify EOS token is appended as last token before and after pruning."""
    tokenizer = create_test_tokenizer()

    eos_id_orig = tokenizer.token_to_id("[EOS]")
    encoded_orig = tokenizer.encode("hello world")
    assert encoded_orig.ids[-1] == eos_id_orig

    test_data = ["hello world"]
    data_iter = create_test_data_iterator(test_data)
    pruned = prune_tokenizer(data_iter, tokenizer)
    eos_id_pruned = pruned.token_to_id("[EOS]")
    encoded_pruned = pruned.encode("hello world")
    assert encoded_pruned.ids[-1] == eos_id_pruned


def test_unk_for_unknown_words():
    """Verify UNK token is used for unknown words before and after pruning."""
    tokenizer = create_test_tokenizer()

    unk_id_orig = tokenizer.token_to_id("[UNK]")
    encoded_orig = tokenizer.encode("antidisestablishmentarianism")
    assert unk_id_orig in encoded_orig.ids

    test_data = ["hello world"]
    data_iter = create_test_data_iterator(test_data)
    pruned = prune_tokenizer(data_iter, tokenizer)
    unk_id_pruned = pruned.token_to_id("[UNK]")
    encoded_pruned = pruned.encode("antidisestablishmentarianism")
    assert unk_id_pruned in encoded_pruned.ids

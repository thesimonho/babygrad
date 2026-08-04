import random

import pytest

from babygrad.tokenizers import BPETokenizer

# Hand-computable corpus: 6 distinct words at fixed frequencies, chosen so the
# pair counts are small enough to tally on paper and so a top-level tie occurs
# early (three pairs reach 10 on round 2, exercising the tie-break).
WORD_FREQUENCIES = {
    "seven": 5,
    "seventeen": 2,
    "seventy": 3,
    "six": 4,
    "sixteen": 2,
    "sixty": 3,
}

CORPUS = " ".join(word for word, freq in WORD_FREQUENCIES.items() for _ in range(freq))

# 8 distinct characters (s e v n t y i x) + `</w>` + 4 specials = 13 base entries.
BASE_VOCAB_SIZE = 13


def _fitted(vocab_size=BASE_VOCAB_SIZE + 4):
    """A tokenizer trained on CORPUS. Default size leaves room for 4 merges."""
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=vocab_size)
    return tokenizer


# --- the hand-verified case -------------------------------------------------


def test_merges_are_learned_in_frequency_weighted_order():
    """The exact merge list, in rank order, for a corpus small enough to tally by hand.

    This is the only test that pins the algorithm itself rather than a property
    of it. Every other test here would still pass on a subtly wrong
    implementation.
    """
    tokenizer = _fitted()

    expected_merges = [
        # 14 = 5 (seven) + 2x2 (seventeen) + 3 (seventy) + 2 (sixteen)
        ("e", "n"),
        # 10, tied with ("s","e") and ("e","v"); tie-break takes the largest pair.
        ("v", "en"),
        # 10, tied with ("e","ven"); "s" > "e".
        ("s", "e"),
        # 10, outright — the corpus has now rebuilt the whole word "seven".
        ("se", "ven"),
    ]

    assert tokenizer.merges == expected_merges


# --- determinism ------------------------------------------------------------


def test_training_twice_gives_an_identical_merge_list():
    """Same corpus, same rules, same result — no dependence on dict iteration luck."""
    first = _fitted()
    second = _fitted()

    assert first.merges == second.merges


def test_merge_list_is_independent_of_word_order_in_the_corpus():
    """The tie-break must decide ties, not insertion order.

    Without an explicit tie-break, `max()` returns the first-encountered maximum,
    so shuffling the corpus silently changes the merge list. This is the test
    that proves the tie-break is doing its job.
    """
    words = CORPUS.split()
    random.Random(0).shuffle(words)

    shuffled = BPETokenizer()
    shuffled.train(" ".join(words), vocab_size=BASE_VOCAB_SIZE + 4)

    assert shuffled.merges == _fitted().merges


# --- structural invariants --------------------------------------------------


def test_vocab_size_is_exactly_what_was_requested():
    """The embedding table is sized off this number, so it must be exact."""
    tokenizer = _fitted(vocab_size=BASE_VOCAB_SIZE + 4)

    assert len(tokenizer.vocab) == BASE_VOCAB_SIZE + 4


def test_each_merge_only_consumes_symbols_that_already_exist():
    """Rank k can only merge symbols produced by the base alphabet or ranks < k.

    A merge referencing a symbol that does not yet exist means the trainer
    recounted against a stale corpus state.
    """
    tokenizer = _fitted()
    known = set(tokenizer.vocab) - {"".join(pair) for pair in tokenizer.merges}

    for left, right in tokenizer.merges:
        assert left in known, f"{left!r} merged before it existed"
        assert right in known, f"{right!r} merged before it existed"
        known.add(left + right)


def test_every_merge_adds_exactly_one_vocabulary_entry():
    """Vocabulary size is base alphabet + specials + one per merge."""
    tokenizer = _fitted(vocab_size=BASE_VOCAB_SIZE + 4)

    assert len(tokenizer.merges) == 4


# --- encode / decode --------------------------------------------------------


def test_roundtrip_for_text_within_the_known_alphabet():
    """Lossless for in-alphabet text. Scoped deliberately: `<unk>` is lossy."""
    tokenizer = _fitted()

    assert tokenizer.decode(tokenizer.encode("seventy six")) == "seventy six"


def test_unseen_word_encodes_via_smaller_units():
    """BPE's whole point: a word absent from the corpus still encodes, using
    known characters and whatever merges happen to apply."""
    tokenizer = _fitted()

    token_ids = tokenizer.encode("vixen")

    assert token_ids  # not empty
    assert tokenizer.decode(token_ids) == "vixen"


def test_unknown_character_becomes_one_unk_per_character():
    """Per-character, not whole-word: `sax` keeps its s and x."""
    tokenizer = _fitted()

    assert tokenizer.decode(tokenizer.encode("sax")) == "s<unk>x"


def test_unknown_characters_are_not_collapsed_together():
    """Two unknown characters produce two `<unk>` tokens, not one."""
    tokenizer = _fitted()

    assert tokenizer.decode(tokenizer.encode("sqqx")) == "s<unk><unk>x"


# --- guards -----------------------------------------------------------------


def test_encoding_before_training_raises():
    """An unfitted tokenizer must fail loudly, not silently emit characters."""
    with pytest.raises(RuntimeError):
        BPETokenizer().encode("seven")


def test_decoding_before_training_raises():
    """Without a vocabulary, ids have no meaning."""
    with pytest.raises(RuntimeError):
        BPETokenizer().decode([0, 1, 2])

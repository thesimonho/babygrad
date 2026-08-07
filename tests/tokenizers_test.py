import random

import pytest

from babygrad.tokenizers import BPETokenizer

# 19 words total. The tokenizer uses the leading-space convention, so every word
# except the very first carries a leading space as part of its symbol tuple.
WORD_FREQUENCIES = {
    "seven": 5,
    "seventeen": 2,
    "seventy": 3,
    "six": 4,
    "sixteen": 2,
    "sixty": 3,
}

CORPUS = " ".join(word for word, freq in WORD_FREQUENCIES.items() for _ in range(freq))

# 9 distinct characters (space s e v n t y i x) + 4 specials = 13 base entries.
# The space counts: under the leading-space convention it is an ordinary symbol.
BASE_VOCAB_SIZE = len(BPETokenizer.SPECIAL_TOKENS) + 9


def _fitted(vocab_size=BASE_VOCAB_SIZE + 4):
    """A tokenizer trained on CORPUS. Default size leaves room for 4 merges."""
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=vocab_size)
    return tokenizer


def _merged(word, pair):
    """Run one merge over a single word and return the resulting symbol tuple."""
    return BPETokenizer()._merge_tokens(word, pair)


def test_a_word_without_the_pair_is_returned_unchanged():
    """The scan must emit every symbol it walks past, including the last one.

    Looping to `len - 1` to keep the lookahead in bounds silently drops the
    final symbol, and a word with no match is the only case that exposes it.
    """
    assert _merged(("c", "a", "t"), ("e", "n")) == ("c", "a", "t")


def test_pair_at_the_start_of_a_word_is_merged():
    assert _merged(("e", "n", "d"), ("e", "n")) == ("en", "d")


def test_pair_in_the_middle_of_a_word_keeps_the_tail():
    """Both failure modes at once: without the skip the `n` is emitted twice,
    and without a full-length scan the trailing `s` is lost."""
    assert _merged(("s", "e", "v", "e", "n", "s"), ("e", "n")) == (
        "s",
        "e",
        "v",
        "en",
        "s",
    )


def test_pair_at_the_very_end_of_a_word_is_merged():
    """Passes even on an implementation that drops the last symbol, because the
    merge consumes it anyway. Kept as a regression case, not as evidence."""
    assert _merged(("s", "e", "v", "e", "n"), ("e", "n")) == ("s", "e", "v", "en")


def test_every_occurrence_of_the_pair_is_merged():
    """One pass merges all occurrences, not just the first."""
    assert _merged(("e", "n", "t", "e", "n"), ("e", "n")) == ("en", "t", "en")


def test_overlapping_pairs_are_merged_greedily_left_to_right():
    """`aaa` merging `aa` yields `aa` + `a`, never `a` + `aa`.

    Encoding replays these merges over new text with the same scan, so the rule
    only has to be consistent — but it does have to be pinned.
    """
    assert _merged(("a", "a", "a"), ("a", "a")) == ("aa", "a")


def test_a_single_symbol_word_is_returned_unchanged():
    """A one-character word has no pair to check. The lookahead must not raise."""
    assert _merged(("a",), ("e", "n")) == ("a",)


def test_merging_an_empty_word_returns_an_empty_tuple():
    """The regex can hand back a zero-length symbol tuple; the scan must not raise."""
    assert _merged((), ("e", "n")) == ()


# --- the hand-verified case -------------------------------------------------


def test_merges_are_learned_in_frequency_weighted_order():
    """The exact merge list, in rank order, for a corpus small enough to tally by hand.

    This is the only test that pins the algorithm itself rather than a property
    of it. Every other test here would still pass on a subtly wrong
    implementation.
    """
    tokenizer = _fitted()

    expected_merges = [
        # 18, outright — every word but the first carries a leading space.
        (" ", "s"),
        # 14 = 5 (seven) + 2x2 (seventeen) + 3 (seventy) + 2 (sixteen)
        ("e", "n"),
        # 10, tied with ("e","v"), which is also 10. Both come from the same
        # four word forms, so nothing separates them on frequency, and the tie
        # falls to the larger pair: ("v","en") > ("e","v") on the first symbol.
        ("v", "en"),
        # 10, outright — completes "even" either way; only the route differs.
        ("e", "ven"),
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

    Without an explicit tie-break the winner falls out of counting order, which
    tracks the order the words appear in, so shuffling the corpus silently
    changes the merge list.
    """
    expected = _fitted().merges

    for seed in range(12):
        words = CORPUS.split()
        random.Random(seed).shuffle(words)

        shuffled = BPETokenizer()
        shuffled.train(" ".join(words), vocab_size=BASE_VOCAB_SIZE + 4)

        assert shuffled.merges == expected, f"merge list changed for seed {seed}"


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


# --- special tokens ---------------------------------------------------------


def test_special_tokens_hold_their_reserved_ids():
    """Collation writes `PAD_ID` into batches and the loss masks it back out, so
    the id has to mean pad regardless of corpus or vocabulary size."""
    tokenizer = _fitted()

    assert tokenizer.vocab[BPETokenizer.PAD_ID] == BPETokenizer.PAD
    assert tokenizer.vocab[BPETokenizer.UNK_ID] == BPETokenizer.UNK
    assert tokenizer.vocab[BPETokenizer.BOS_ID] == BPETokenizer.BOS
    assert tokenizer.vocab[BPETokenizer.EOS_ID] == BPETokenizer.EOS


def test_special_ids_do_not_move_with_the_corpus():
    """A different alphabet must not shift the reserved block."""
    other = BPETokenizer()
    other.train("abcabc", vocab_size=len(BPETokenizer.SPECIAL_TOKENS) + 3)

    assert other.vocab[BPETokenizer.PAD_ID] == BPETokenizer.PAD
    assert other.vocab[BPETokenizer.EOS_ID] == BPETokenizer.EOS


def test_padding_is_not_confused_with_an_unknown_character():
    """The bug this reserved block exists to prevent: pad and unk sharing an id
    would make padding indistinguishable from an unrepresentable character."""
    assert BPETokenizer.PAD_ID != BPETokenizer.UNK_ID


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

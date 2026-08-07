import re
from abc import ABC, abstractmethod
from collections import Counter


class Tokenizer(ABC):
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPECIAL_TOKENS = (PAD, UNK, BOS, EOS)

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self):
        self.vocab = []
        self.merges: list[tuple[str, str]] = []

    @abstractmethod
    def train(self, corpus: str, vocab_size: int):
        pass

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        pass


class BPETokenizer(Tokenizer):
    """Byte pair encoding"""

    word_template = re.compile(
        r"'(?i:t|ll|d|s|re|ve|m)"  # contraction suffixes
        r"|\s?[^\W\d_]+"  # optional space + letters (unicode-aware)
        r"|\s?[0-9]+"  # optional space + digits
        r"|\s?[^\s\w]+"  # optional space + punctuation/symbols
        r"|\s?_+"  # optional space + underscores
        r"|\s+(?!\S)"  # whitespace run, leaving one space for the next word
        r"|\s+"  # any remaining whitespace
    )

    def train(self, corpus: str, vocab_size: int):
        alphabet = sorted(set(corpus))
        self.vocab = list(self.SPECIAL_TOKENS) + alphabet

        words: list[str] = self.word_template.findall(corpus)
        split_word_counts = Counter(tuple(word) for word in words)

        while len(self.vocab) < vocab_size:
            bigram_count: Counter[tuple[str, str]] = Counter()
            # count bigram frequency across all words
            for letters, count in split_word_counts.items():
                for i in range(len(letters) - 1):
                    pair = (letters[i], letters[i + 1])
                    bigram_count[pair] += count

            # store the most frequent bigram; ties break on the pair itself so
            # the merge list depends only on the corpus, not on the order its
            # words happened to be counted in
            if not bigram_count:
                break
            most_common_pair = max(bigram_count, key=lambda p: (bigram_count[p], p))
            self.vocab.append("".join(most_common_pair))
            self.merges.append(most_common_pair)

            # do merges
            new_split_word_counts = Counter()
            for tokens in split_word_counts.keys():
                merged = self._merge_tokens(tokens, most_common_pair)
                new_split_word_counts[merged] += split_word_counts[tokens]

            split_word_counts = new_split_word_counts.copy()

    def encode(self, string: str) -> list[int]:
        """Turn text into a list of token ids."""
        if not self.vocab:
            raise RuntimeError("Model is not trained")

        encoded = []

        words: list[str] = self.word_template.findall(string)

        for word in words:
            for merge in self.merges:
                word = self._merge_tokens(tuple(word), merge)

            for token in word:
                try:
                    encoded.append(self.vocab.index(token))
                except ValueError:
                    encoded.append(self.UNK_ID)

        return encoded

    def decode(self, token_ids: list[int]) -> str:
        """Turn a list of token ids into a string of concatenated tokens."""
        if not self.vocab:
            raise RuntimeError("Model is not trained")

        output = ""
        for id in token_ids:
            output += self.vocab[id]

        return output

    def _merge_tokens(
        self, tokens: tuple[str, ...], merge_pair: tuple[str, str]
    ) -> tuple[str, ...]:
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == merge_pair[0]
                and tokens[i + 1] == merge_pair[1]
            ):
                merged_token = "".join(merge_pair)
                merged_tokens.append(merged_token)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return tuple(merged_tokens)

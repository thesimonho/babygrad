from abc import ABC, abstractmethod
from collections import Counter


class Tokenizer(ABC):
    def __init__(self):
        self.vocab: dict[str, int] = {}
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

    def train(self, corpus: str, vocab_size: int):
        word_counts: Counter[str] = Counter()
        for word in corpus.split(" "):
            word_counts[word] += 1

        temp_vocab: Counter[tuple[str, str]] = Counter()
        while len(self.vocab) < vocab_size:
            for word, count in word_counts.items():
                letters: list[str] = list(word) + ["\\w"]

                for i in range(len(letters) - 1):
                    pair = (letters[i], letters[i + 1])
                    temp_vocab[pair] += count

            most_common_pair, most_common_count = temp_vocab.most_common(1)[0]
            self.vocab["".join(most_common_pair)] = most_common_count
            self.merges.append(most_common_pair)
            print(self.vocab)
            print(self.merges)

            for word, count in word_counts.items():
                letters: list[str] = list(word) + ["\\w"]
                new_letters = []
                for i in range(len(letters) - 1):
                    if (
                        letters[i] == most_common_pair[0]
                        and letters[i + 1] == most_common_pair[1]
                    ):
                        new_letters.append("".join(most_common_pair))
                    else:
                        new_letters.append(letters[i])

            break

        # end by generating a list of ordered tokens for quicker lookup
        self.tokens_by_id = sorted(self.vocab, key=lambda token: self.vocab[token])

    def encode(self, string: str) -> list[int]:
        if not self.vocab:
            raise RuntimeError("Model is not trained")

        output = []

        return output

    def decode(self, token_ids: list[int]) -> str:
        if not self.vocab:
            raise RuntimeError("Model is not trained")

        output = ""
        for id in token_ids:
            output += self.tokens_by_id[id]

        return output

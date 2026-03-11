from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from pathlib import Path
from typing import List, Union, Optional, Dict
import pickle
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GeneVocab:
    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], "GeneVocab"],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ):
        if isinstance(gene_list_or_vocab, GeneVocab):
            if specials:
                raise ValueError(
                    "Cannot pass specials when initializing from GeneVocab."
                )
            self.tokenizer = gene_list_or_vocab.tokenizer
        elif isinstance(gene_list_or_vocab, list):
            specials = specials or []
            vocab_tokens = list(gene_list_or_vocab)

            # Insert specials at correct position
            for tok in reversed(specials):
                if tok not in vocab_tokens:
                    if special_first:
                        vocab_tokens.insert(0, tok)
                    else:
                        vocab_tokens.append(tok)

            trainer = trainers.WordLevelTrainer(special_tokens=specials)
            self.tokenizer = Tokenizer(models.WordLevel(unk_token=None))
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            self.tokenizer.train_from_iterator(
                [[token] for token in vocab_tokens], trainer=trainer
            )
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of tokens or a GeneVocab instance."
            )

        self.default_token = default_token
        if default_token is not None and default_token in self.get_stoi():
            self.set_default_token(default_token)

    def __call__(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self[tokens]
        return [self[token] for token in tokens]

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> "GeneVocab":
        # Sort tokens by index to preserve ordering
        sorted_tokens = sorted(token2idx.items(), key=lambda x: x[1])
        vocab_tokens = [token for token, _ in sorted_tokens]

        # Build WordLevel model directly with provided vocab
        tokenizer = Tokenizer(models.WordLevel(vocab=token2idx, unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        vocab = cls([])
        vocab.tokenizer = tokenizer
        vocab.set_default_token(default_token)

        return vocab

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "GeneVocab":
        file_path = Path(file_path)
        if file_path.suffix == ".json":
            tokenizer = Tokenizer.from_file(str(file_path))
            vocab = cls([])
            vocab.tokenizer = tokenizer
            return vocab
        elif file_path.suffix == ".pkl":
            with open(file_path, "rb") as f:
                tokenizer = pickle.load(f)
            vocab = cls([])
            vocab.tokenizer = tokenizer
            return vocab
        else:
            raise ValueError("Only .json and .pkl formats are supported.")

    def save_json(self, file_path: Union[str, Path]) -> None:
        self.tokenizer.save(str(file_path))

    def __getitem__(self, token: str) -> int:
        stoi = self.get_stoi()
        if token in stoi:
            return stoi[token]
        elif self.default_token and self.default_token in stoi:
            return stoi[self.default_token]
        else:
            raise KeyError(f"{token} not found in vocabulary.")

    def __contains__(self, token: str) -> bool:
        return token in self.get_stoi()

    def __len__(self) -> int:
        return len(self.get_stoi())

    def token_to_id(self, token: str) -> int:
        return self[token]

    def id_to_token(self, idx: int) -> str:
        itos = self.get_itos()
        if 0 <= idx < len(itos):
            return itos[idx]
        else:
            raise IndexError(f"Index {idx} out of range.")

    def get_stoi(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_itos(self) -> List[str]:
        return [
            token for token, _ in sorted(self.get_stoi().items(), key=lambda x: x[1])
        ]

    def set_default_token(self, default_token: str):
        if default_token not in self:
            raise ValueError(f"Default token '{default_token}' not in vocabulary.")
        self.default_token = default_token

import os
import json
from collections import Counter
from string import punctuation
from urllib import request
from zipfile import ZipFile
from typing import Any, Dict, List, NamedTuple, Optional, Union, Set

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset



class Movie(NamedTuple):
    budget: int
    genres: List[str]
    homepage: str
    id: str
    keywords: List[str]
    original_language: str
    original_title: str
    overview: str
    popularity: float
    # production_companies: List[str]
    # production_countries: List[str]
    release_date: str
    revenue: int
    # runtime: int
    # spoken_languages: List[str]
    # status: str
    # tagline: str
    title: str
    # vote_average: float
    # vote_count: int
    overview_tokens: Optional[List[str]] = None
    overview_indices: Optional[List[str]] = None
    genres_indices: Optional[List[str]] = None

    @staticmethod
    def from_raw(raw_input: Dict[str, str]) -> "Movie":
        assert len(raw_input) == 20

        fields: Dict[str, Any] = {}
        fields["budget"] = int(raw_input["budget"])
        fields["genres"] = [genre["name"] for genre in json.loads(raw_input["genres"])]
        fields["homepage"] = raw_input["homepage"]
        fields["id"] = raw_input["id"]
        fields["keywords"] = [kw["name"] for kw in json.loads(raw_input["keywords"])]
        fields["original_language"] = raw_input["original_language"]
        fields["original_title"] = raw_input["original_title"]
        fields["overview"] = raw_input["overview"]
        fields["popularity"] = raw_input["popularity"]

        # TODO: Add production_companies and production_countries

        fields["release_date"] = raw_input["release_date"]
        fields["revenue"] = int(raw_input["revenue"])
        # fields["runtime"] = int(raw_input[13])
        fields["title"] = raw_input["title"]

        return Movie(**fields)


def tokenize(text: str, remove_punct: bool = True, mode: str = "simple") -> List[str]:
    """Splits text into tokens

    Args:
        text (str): a text to tokenize
        remove_punct (bool, optional): whether to remove puntuation . Defaults to True.
        mode (str, optional): How to tokenize a text.
            Currently supported options are "simple" (str.split()) and "spacy".
            Defaults to "simple".

    Returns:
        List[str]: A list of tokens
    """
    if mode == "simple":
        if remove_punct:
            text = "".join([c.lower() for c in text if c not in punctuation])
        return text.split()

    if mode == "spacy":
        from spacy.lang.en import English

        nlp = English()
        tokens = nlp(text)
        if remove_punct:
            return [token.text for token in tokens if not token.is_punct]
        return [token.text for token in tokens]


class Vocab:
    def __init__(
        self,
        counter: Counter,
        min_freq: int = 1,
        max_words: int = None,
        specials: List[str] = ["<PAD>", "<UNK>"],
    ) -> None:
        """Represents a vocabulary object

        Args:
            counter (Counter): collections.Counter containing counts of tokens in a corpus
            min_freq (int, optional): Minimum frequency of a token to be added to Vocab.
                Defaults to 1.
            max_words (int, optional): Maximum number of tokens.
                If None all words will be added. Defaults to None.
            specials (List[str], optional): A list of additional tokens to add.
                Defaults to ["<PAD>"].
        """
        self.counter = counter
        self.min_freq = min_freq
        self.max_words = max_words

        self.word2ix: Dict[str, int] = {}
        self.ix2word: Dict[int, str] = {}

        for ix, word in enumerate(specials):
            self.word2ix[word] = ix
            self.ix2word[ix] = word

        for ix, (word, freq) in enumerate(
            self.counter.most_common(max_words), start=len(self.word2ix)
        ):
            self.word2ix[word] = ix
            self.ix2word[ix] = word

            if freq < self.min_freq:
                break

    def __getitem__(self, item: Union[str, int]) -> Union[str, int]:
        if isinstance(item, str):
            if item in self.word2ix:
                return self.word2ix[item]
            return self.word2ix["<UNK>"]  # handling unknown

        if isinstance(item, int):
            return self.ix2word[item]

        raise KeyError("Key should be either str or int")

    def __len__(self) -> int:
        return len(self.word2ix)

    def __contains__(self, word):
        return word in self.word2ix

    def set_vectors(self, vectors: Dict[str, List[float]], dim: int = 300) -> None:
        """Sets vectors for tokens that exist in Vocab

        Args:
            vectors (Dict[str, List[float]]): A dictionary with vectors
            dim (int, optional): Dimensionality of vectors. Defaults to 300.
        """
        self.vectors = torch.zeros(len(self.ix2word), dim)
        for ix, word in self.ix2word.items():
            vec = vectors.get(word)
            if vec:
                self.vectors[ix] = torch.tensor(vec)


def load_fasttext(fname, keep_vectors: Set[str]) -> Dict[str, List[float]]:
    """Loads vectors from a zipped file or downloads fasttext first if file does not exist

    Args:
        fname ([type]): zipped file with vectors
        keep_vectors (Set[str]): a set of tokens to keep vectors for

    Returns:
        Dict[str, List[float]]: [description]
    """
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"

    if not os.path.exists(fname):
        print("Downloading vectors...")
        request.urlretrieve(url, fname)
    zipped = ZipFile(fname)

    with zipped.open("wiki-news-300d-1M.vec", "r") as fin:
        n, d = map(int, fin.readline().decode("utf-8").split())
        data = {}
        for line in tqdm(fin, total=n):
            tokens = line.decode("utf-8").rstrip().split(" ")
            if tokens[0] in keep_vectors:
                data[tokens[0]] = [float(val) for val in tokens[1:]]

        return data


class MovieDataset(Dataset):
    def __init__(
        self,
        movies: List[Movie],
        device: str,
        num_classes: int,
        max_len: int = 100,
        pad_index: int = 0,
    ):
        self.movies_list = movies
        self.max_len = max_len
        self.pad_index = pad_index
        self.num_classes = num_classes
        self.device = device

    def __len__(self):
        return len(self.movies_list)

    def __getitem__(self, idx):
        X = torch.LongTensor(self.movies_list[idx].overview_indices[: self.max_len])
        X_len = torch.ShortTensor([X.shape[0]]).squeeze()
        X = F.pad(X, (0, self.max_len - X.shape[0])).to(self.device)
        y = torch.FloatTensor(
            [
                ix in self.movies_list[idx].genres_indices
                for ix in range(self.num_classes)
            ]
        ).to(self.device)
        sample = {"idx": idx, "X": X, "X_len": X_len, "y": y}
        return sample

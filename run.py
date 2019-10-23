import csv
import io

from collections import Counter
from itertools import chain
from zipfile import ZipFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataloader import Movie, MovieDataset, Vocab, load_fasttext, tokenize
from models import SimpleLSTM
from utils import train


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on : {device}")
SEED = 22

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Data preprocessing
zipped = ZipFile("data/tmdb-5000-movie-dataset.zip")
with zipped.open("tmdb_5000_movies.csv") as file:
    reader = csv.DictReader(io.TextIOWrapper(file, "utf-8"))
    # # skip header
    # _ = next(reader)
    movies = [Movie.from_raw(line) for line in reader]


# Data cleanup
genres_to_remove = ["TV Movie", "Foreign"]

for movie in movies:
    for genre in genres_to_remove:
        if genre in movie.genres:
            movie.genres.remove(genre)

movies = [movie for movie in movies if movie.overview.strip() and movie.genres]

print(f"There are {len(movies)} movies left")

movies = [
    movie._replace(
        overview_tokens=tokenize(movie.overview, remove_punct=False, mode="simple")
    )
    for movie in movies
]


all_overviews = chain.from_iterable([movie.overview_tokens for movie in movies])
counter = Counter(all_overviews)
overviews_vocab = Vocab(counter)

all_genres = chain.from_iterable([movie.genres for movie in movies])
genres_counter = Counter(all_genres)
genres_vocab = Vocab(genres_counter)


# Numericalizing fields

movies = [
    movie._replace(
        overview_indices=[overviews_vocab[word] for word in movie.overview_tokens],
        genres_indices=[genres_vocab[genre] for genre in movie.genres],
    )
    for movie in movies
]


vectors = load_fasttext(
    "wiki-news-300d-1M.vec.zip", set(overviews_vocab.word2ix.keys())
)

overviews_vocab.set_vectors(vectors, dim=300)


train_len = round(len(movies) * 0.9)
ds = MovieDataset(movies, device, num_classes=len(genres_vocab), max_len=300)

ds_train, ds_valid = random_split(ds, lengths=[train_len, len(movies) - train_len])


if __name__ == "__main__":
    BATCH_SIZE = 32
    HIDDEN_DIM = 128
    BIDIRECTIONAL = True
    NUM_LAYERS = 1
    DROPOUT = 0.1
    N_EPOCHS = 1
    PRINT_EVERY = 1

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleLSTM(
        n_out=len(genres_vocab),
        vocab_size=len(overviews_vocab),
        vectors=overviews_vocab.vectors,
        seq_len=300,
        hidden_dim=HIDDEN_DIM,
        bidirectional=BIDIRECTIONAL,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train(
        model,
        dl_train,
        dl_valid,
        criterion,
        optimizer,
        n_epochs=N_EPOCHS,
        print_every=PRINT_EVERY,
    )
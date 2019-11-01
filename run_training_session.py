import csv
import io
import os
import pickle

from collections import Counter
from itertools import chain
from zipfile import ZipFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import mlflow

from dataloader import Movie, MovieDataset, Vocab, load_fasttext, tokenize
from models import SimpleLSTM, SimpleCNN, ConcatPoolLSTM
from utils import train


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")
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
    "wiki-news-300d-1M.vec", set(overviews_vocab.word2ix.keys()), directory="vectors"
)

overviews_vocab.set_vectors(vectors, dim=300)


train_len = round(len(movies) * 0.9)
ds = MovieDataset(movies, device, num_classes=len(genres_vocab), max_len=300)

ds_train, ds_valid = random_split(ds, lengths=[train_len, len(movies) - train_len])


if __name__ == "__main__":


    # TODO: Make these agrparse parameters
    MODEL_TYPE = "lstm"
    SAVE_MODEL = True

    N_EPOCHS = 1
    BATCH_SIZE = 32
    HIDDEN_DIM = 128
    BIDIRECTIONAL = False
    NUM_LAYERS = 1
    DROPOUT = 0.1
    PRINT_EVERY = 1
    NUM_FILTERS = 12
    FILTER_SIZES = [1, 3, 5]
    WDROP = 0.2

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=True)

    model_classes = {
        "lstm": SimpleLSTM,
        "lstm_pooling": ConcatPoolLSTM,
        "cnn": SimpleCNN
    }

    if MODEL_TYPE in ("lstm", "lstm_pooling"):
        model_args = {
            "n_out": len(genres_vocab),
            "vocab_size": len(overviews_vocab),
            "vectors": overviews_vocab.vectors,
            "seq_len": 300,
            "hidden_dim": HIDDEN_DIM,
            "bidirectional": BIDIRECTIONAL,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "wdrop": WDROP,
        }
    elif MODEL_TYPE == "cnn":
        model_args = {
            "n_out": len(genres_vocab),
            "vectors": overviews_vocab.vectors,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "num_filters": NUM_FILTERS,
            "filter_sizes": FILTER_SIZES,
        }

    model_class = model_classes[MODEL_TYPE]

    model = model_class(**model_args).to(device)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train(
        model,
        MODEL_TYPE,
        dl_train,
        dl_valid,
        criterion,
        optimizer,
        n_epochs=N_EPOCHS,
        print_every=PRINT_EVERY,
        logger=mlflow.log_metrics
    )

    if SAVE_MODEL:
        if not os.path.exists("outputs"):
            os.mkdir("outputs")

        if not os.path.exists("outputs/models"):
            os.mkdir("outputs/models")

        if not os.path.exists("outputs/vocab"):
            os.mkdir("outputs/vocab")

        serialization_dictionary = {
            "model_weights": model.state_dict(),
            "model_args": model_args
        }

        torch.save(serialization_dictionary, "outputs/models/{0}_{1}.pth".format(MODEL_TYPE, N_EPOCHS))

        with open("outputs/vocab/overviews_vocab.pcl", "wb") as f:
            pickle.dump(overviews_vocab, f)
        with open("outputs/vocab/genres_vocab.pcl", "wb") as f:
            pickle.dump(genres_vocab, f)

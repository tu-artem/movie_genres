import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SimpleLSTM
from dataloader import Vocab, tokenize


def make_predictions(
    model: nn.Module,
    vocab: Vocab,
    text: str,
    max_len: int = 300,
    device: str = "cpu",
):

    tokens = tokenize(text, remove_punct=False, mode="simple")
    tokens_idx = [vocab[token] for token in tokens]
    X = torch.LongTensor(tokens_idx[:max_len])
    X_len = torch.ShortTensor([X.shape[0]])
    X = F.pad(X, (0, max_len - X.shape[0])).to(device)
    X = X.unsqueeze(0)

    predictions = model(X, X_len)
    return predictions


device = "cuda" if torch.cuda.is_available() else "cpu"

model_dictionary = torch.load("outputs/models/simple_lstm_10.pth", map_location=device)

model = SimpleLSTM(**model_dictionary["model_args"])

model.load_state_dict(model_dictionary["model_weights"])
model.to(device)
model.eval()

with open("outputs/vocab/overviews_vocab.pcl", "rb") as f:
    overviews_vocab = pickle.load(f)
with open("outputs/vocab/genres_vocab.pcl", "rb") as f:
    genres_vocab = pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="This is some funny story")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--interactive", action='store_true')


    args = parser.parse_args()
    threshold = args.threshold
    interactive = args.interactive
    if interactive:
        while True:
            text = input("Enter text to score: \n")
            if not text:
                continue
            if text == "exit":
                break
            predictions = make_predictions(model, vocab=overviews_vocab, text=text)
            genre_indices = (torch.sigmoid(predictions.squeeze()) > threshold).nonzero().squeeze(0).tolist()

            predicted_genres = [genres_vocab[ix] for ix in genre_indices]
            print(f'Predicted: {", ".join(predicted_genres)}')
            print("Type exit to finish interactive session\n")
    else:
        text = args.text

        predictions = make_predictions(model, vocab=overviews_vocab, text=text)
        genre_indices = (torch.sigmoid(predictions.squeeze()) > threshold).nonzero().squeeze(0).tolist()

        predicted_genres = [genres_vocab[ix] for ix in genre_indices]
        print(text)
        print(f'Predicted: {", ".join(predicted_genres)}')

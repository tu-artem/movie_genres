import torch

from tqdm import tqdm


def multilabel_accuracy(
    y_true: torch.Tensor, y_pred: torch.Tensor, threshhold: float = 0.5
) -> float:
    """Calculates accuracy of a multilabel classification as intersection over union of
    predicted and correct classes

    Args:
        y_true (torch.Tensor): Correct classes
        y_pred (torch.Tensor): Predicted probabilities
        threshhold (float, optional): Where to cut predicted probabilities. Defaults to 0.5.

    Returns:
        float: [description]
    """
    pred_labels = y_pred > threshhold
    intersect = pred_labels & y_true.bool()
    union = pred_labels | y_true.bool()
    acc = intersect.sum(dim=1).float() / union.sum(dim=1).float()
    return acc.mean().item()


def train(model, dl_train, dl_valid, criterion, optimizer, n_epochs, print_every, logger=None):
    train_losses = []
    valid_losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        valid_epoch_loss = 0.0
        epoch_acc = 0.0
        valid_epoch_acc = 0.0
        model.train()
        for sample in tqdm(dl_train):

            model.zero_grad()

            predicted = model(sample["X"], sample["X_len"])
            loss = criterion(predicted, sample["y"])
            epoch_loss += loss.item()

            epoch_acc += multilabel_accuracy(sample["y"], torch.sigmoid(predicted))

            loss.backward()
            optimizer.step()
        # validation
        for sample in dl_valid:
            model.eval()
            predicted = model(sample["X"], sample["X_len"])
            loss = criterion(predicted, sample["y"])
            valid_epoch_loss += loss.item()
            valid_epoch_acc += multilabel_accuracy(sample["y"], torch.sigmoid(predicted))
        train_losses.append(epoch_loss / len(dl_train))
        valid_losses.append(valid_epoch_loss / len(dl_valid))
        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch + 1} "
                f"Train Loss {(epoch_loss / len(dl_train)):.4f}, "
                f"Valid Loss {(valid_epoch_loss / len(dl_valid)):.4f}, "
                f"Train Acc {(epoch_acc / len(dl_train)):.4f}, "
                f"Valid Acc {(valid_epoch_acc / len(dl_valid)):.4f}"
            )

        metrics = {
            "train_loss": epoch_loss / len(dl_train),
            "valid_loss": valid_epoch_loss / len(dl_valid),
            "train_acc": epoch_acc / len(dl_train),
            "valid_acc": valid_epoch_acc / len(dl_valid)
        }

        if logger:
            logger(metrics)

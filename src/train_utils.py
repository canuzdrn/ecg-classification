from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scheduler=None):
    """
    training function that's used in the training loop
    enhanced modularity by writing an additional function for one epoch training
    """
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for signals, labels, lengths in dataloader:
        # move to device (cpu by default)
        signals = [s.to(device) for s in signals]
        labels = labels.to(device)

        # forwards pass
        outputs = model(signals, lengths)
        loss = loss_fn(outputs, labels)

        # backward pass (backprop)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # step the OneCycleLR scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # update metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1

def evaluate(model, dataloader, loss_fn, device):
    """
    basically evaluation/inference function
    used to evaluate validation metrics after each epoch (e.g. early stopping after finding best model)
    """

    # enter into eval mode
    model.eval()

    total_loss = 0
    all_preds, all_labels = [], []

    # inference
    with torch.no_grad():
        for signals, labels, lengths in dataloader:
            signals = [s.to(device) for s in signals]
            labels = labels.to(device)

            outputs = model(signals, lengths)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")  # using macro average for class balance

    # uncomment below line to log -- observe whether there is any one-class collapse situation
    #print("Unique predictions:", np.unique(all_preds, return_counts=True))

    return avg_loss, acc, f1

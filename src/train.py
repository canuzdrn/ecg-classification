import time
from src.train_utils import train_one_epoch, evaluate
import torch

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10):
    """
    trains a pytorch model over multiple epochs and evaluates it on a validation set after each epoch.
    
    function tracks the best performing model (based on validation macro F1-score) and returns 
    the model with the best validation performance at the end of training.
    Also returns a history dictionary with metrics for visualization.
    """

    # best f1 so far
    best_f1 = 0
    # best model state (dict) -- None at the start
    best_model_state = None

    # history dictionary to store metrics per epoch
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": []
    }

    # learning rate scheduler -- decrease lr on plateau, let the model decide the lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # training loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_f1)

        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        # Check the updated LR
        for param_group in optimizer.param_groups:
            print("Current learning rate:", param_group["lr"])

        duration = time.time() - start_time

        # logs
        print(f"Epoch {epoch:02d} | Time: {duration:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()

    # Load best model (exited loop)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
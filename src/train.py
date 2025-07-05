import time
from src.train_utils import train_one_epoch, evaluate
import torch
from torch.optim.lr_scheduler import OneCycleLR
from src.train_utils import evaluate

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

    # ONE‐CYCLE LR Scheduler preferred since it adapts learning rate dynamically
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,                    # peak LR
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,                  # 30% of cycle spent ramping up
        anneal_strategy="cos",          # cosine annealing down
        div_factor=25.0,                # initial LR = max_lr / div_factor
        final_div_factor=1e4            # final LR = max_lr / final_div_factor
    )

    # training loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # pass scheduler into train_one_epoch so it can step per batch
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scheduler
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, device)

        # Save metrics to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

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

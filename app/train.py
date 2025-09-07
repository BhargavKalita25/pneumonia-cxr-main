import os, json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim
from app.config import TrainConfig, load_profile
from app.data.dataset import CXRFolder
from app.models.builder import build_model
from app.models.losses import weighted_bce_with_logits, focal_loss_with_logits
from app.utils.seed import set_seed
from app.utils.metrics import compute_all_metrics
from app.utils.plots import plot_training_curves


def get_pos_weight(ds):
    """Compute positive class weight for imbalanced datasets."""
    labels = [ds[i][1].item() for i in range(len(ds))]
    pos, neg = sum(labels), len(labels) - sum(labels)
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, scaler, device, loss_name, pos_weight):
    """Train model for one epoch."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y, _, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            if loss_name == "focal":
                loss = focal_loss_with_logits(logits, y)
            else:
                loss = weighted_bce_with_logits(
                    logits, y, pos_weight.to(device) if pos_weight is not None else None
                )

        # backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # accuracy
        prob = torch.sigmoid(logits).squeeze(1)
        pred = (prob >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()

        # accumulate loss
        loss_sum += loss.item() * y.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, device, loss_name="bce", pos_weight=None):
    """Validate model on a dataset and return metrics + val_loss."""
    model.eval()
    y_true, y_prob, val_loss_sum, total = [], [], 0.0, 0

    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu()
        y_true += y.cpu().numpy().tolist()
        y_prob += prob.numpy().tolist()

        # compute validation loss
        if loss_name == "focal":
            loss = focal_loss_with_logits(logits, y)
        else:
            loss = weighted_bce_with_logits(
                logits, y, pos_weight.to(device) if pos_weight is not None else None
            )
        val_loss_sum += loss.item() * y.size(0)
        total += y.size(0)

    metrics = compute_all_metrics(y_true, y_prob)
    metrics["val_loss"] = val_loss_sum / total
    return metrics


def main():
    cfg = load_profile(TrainConfig())
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # datasets
    train_ds = CXRFolder(cfg.data_root, "train", cfg.img_size, True)
    val_ds = CXRFolder(cfg.data_root, "val", cfg.img_size, False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # model
    model = build_model(cfg.model_name, 1, pretrained=True).to(device)

    # freeze backbone (transfer learning warmup)
    if cfg.freeze_backbone_epochs > 0:
        for n, p in model.named_parameters():
            if not ("classifier" in n or "fc" in n):
                p.requires_grad = False

    # optimizer + scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.scheduler == "cosine"
        else optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, factor=0.5
        )
    )

    # AMP + class imbalance
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device == "cuda"))
    pos_weight = None if cfg.class_weight == 0 else torch.tensor([cfg.class_weight])
    if cfg.class_weight == 0:
        pos_weight = get_pos_weight(train_ds)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_auc, best_path = -1, os.path.join(cfg.output_dir, "best.ckpt")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        # unfreeze backbone after warmup
        if epoch == cfg.freeze_backbone_epochs:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

        # training
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg.loss, pos_weight
        )

        # validation
        metrics = validate(model, val_loader, device, cfg.loss, pos_weight)
        val_loss, val_auc, val_acc = (
            metrics["val_loss"],
            metrics["roc_auc"],
            metrics["accuracy"],
        )

        if cfg.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_auc)

        # save history
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss)  # ✅ now saved
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {val_loss:.4f} AUC {val_auc:.3f} acc {val_acc:.3f}"
        )

        # save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, best_path)
            with open(
                os.path.join(cfg.output_dir, "val_metrics.json"), "w"
            ) as f:
                json.dump(metrics, f, indent=2)
        else:
            patience_counter += 1

        # early stopping
        if patience_counter >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # plot training history
    plot_training_curves(hist, cfg.output_dir)
    print(f"Best val ROC-AUC {best_auc:.3f} saved at {best_path}")


if __name__ == "__main__":
    main()

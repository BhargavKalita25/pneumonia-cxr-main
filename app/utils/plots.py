import os, matplotlib.pyplot as plt, numpy as np

def plot_training_curves(hist, outdir):
    os.makedirs(outdir, exist_ok=True)
    # loss
    plt.figure()
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss")
    plt.legend(); plt.savefig(os.path.join(outdir,"loss.png")); plt.close()

    # accuracy
    plt.figure()
    plt.plot(hist["train_acc"], label="train")
    plt.plot(hist["val_acc"], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy")
    plt.legend(); plt.savefig(os.path.join(outdir,"acc.png")); plt.close()

def plot_roc_pr(metrics, outdir):
    fpr, tpr = metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"]
    prec, rec = metrics["pr_curve"]["precision"], metrics["pr_curve"]["recall"]

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC AUC={metrics['roc_auc']:.3f}")
    plt.savefig(os.path.join(outdir,"roc.png")); plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR AUC={metrics['pr_auc']:.3f}")
    plt.savefig(os.path.join(outdir,"pr.png")); plt.close()

def plot_calibration(y_true, y_prob, outdir, bins=10):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    qs = np.linspace(0,1,bins+1)
    idx = np.digitize(y_prob, qs)-1
    means = [y_prob[idx==i].mean() if (idx==i).any() else np.nan for i in range(bins)]
    fracs = [y_true[idx==i].mean() if (idx==i).any() else np.nan for i in range(bins)]
    plt.figure()
    plt.plot([0,1],[0,1],'--')
    plt.scatter(means, fracs)
    plt.xlabel("Predicted prob"); plt.ylabel("Observed freq")
    plt.title("Calibration")
    plt.savefig(os.path.join(outdir,"calibration.png")); plt.close()

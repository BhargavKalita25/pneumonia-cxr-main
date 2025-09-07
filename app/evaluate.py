import os, json, torch
from torch.utils.data import DataLoader
from app.config import TrainConfig, load_profile
from app.data.dataset import CXRFolder
from app.models.builder import build_model
from app.utils.metrics import compute_all_metrics
from app.utils.plots import plot_roc_pr, plot_calibration

def main():
    cfg = load_profile(TrainConfig())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_ds = CXRFolder(cfg.data_root,"test",cfg.img_size,False)
    test_loader = DataLoader(test_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)

    ckpt = torch.load(os.path.join(cfg.output_dir,"best.ckpt"),map_location=device)
    model = build_model(cfg.model_name,1).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    y_true,y_prob=[],[]
    with torch.no_grad():
        for x,y,_,_ in test_loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).squeeze(1).cpu().numpy().tolist()
            y_prob+=p; y_true+=y.numpy().tolist()

    metrics = compute_all_metrics(y_true,y_prob)
    os.makedirs(cfg.output_dir,exist_ok=True)
    with open(os.path.join(cfg.output_dir,"test_metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    plot_roc_pr(metrics,cfg.output_dir)
    plot_calibration(y_true,y_prob,cfg.output_dir)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()

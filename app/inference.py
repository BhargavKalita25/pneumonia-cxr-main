import os, torch, numpy as np
from PIL import Image
from torchvision import transforms
from app.models.builder import build_model
from app.models.gradcam import GradCAM, overlay_cam

class Predictor:
    def __init__(self, arch, ckpt_path, img_size=224, device="cpu"):
        # ✅ Ensure checkpoint exists
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. "
                "Please train the model first with: python -m app.train"
            )

        self.device = device
        self.model = build_model(arch, 1, pretrained=False).to(device)

        try:
            state = torch.load(ckpt_path, map_location=device)["model"]
            self.model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}")

        self.model.eval()

        # Preprocessing transform
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # Pick Grad-CAM layer
        if "densenet" in arch:
            self.target_layer = "features.denseblock4"
        elif "resnet" in arch:
            self.target_layer = "layer4"
        else:
            self.target_layer = list(dict([*self.model.named_modules()]).keys())[-1]

        self.cam = GradCAM(self.model, target_layer=self.target_layer)

    def predict_with_cam(self, pil_img, out_path):
        rgb = pil_img.convert("RGB")
        x = self.tfm(rgb).unsqueeze(0).to(self.device)

        try:
            cam, prob = self.cam(x)
        except Exception as e:
            raise RuntimeError(f"Grad-CAM failed: {e}")

        # Resize CAM to match input image size
        cam = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(rgb.size)
        ) / 255.0

        # Overlay CAM
        over = overlay_cam(np.array(rgb), cam, alpha=0.45)
        Image.fromarray(over).save(out_path)

        label = "Pneumonia" if prob >= 0.5 else "Normal"
        conf = prob if prob >= 0.5 else 1 - prob
        return label, float(conf)

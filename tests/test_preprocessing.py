from app.data.dataset import default_transforms
from PIL import Image
import torch

def test_transform_output_shape():
    tfm=default_transforms(224,train=False)
    x=tfm(Image.new("RGB",(512,512)))
    assert x.shape==(3,224,224)
    assert torch.isfinite(x).all()

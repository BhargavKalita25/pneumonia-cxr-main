import numpy as np, pydicom
from PIL import Image
from io import BytesIO

def dicom_to_png_bytes(dcm_bytes):
    ds = pydicom.dcmread(BytesIO(dcm_bytes))
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds,"WindowCenter") and hasattr(ds,"WindowWidth"):
        wc, ww = float(ds.WindowCenter), float(ds.WindowWidth)
        arr = np.clip(arr, wc-ww/2, wc+ww/2)
    arr -= arr.min(); arr /= (arr.max()+1e-6)
    arr = (arr*255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    out = BytesIO(); img.save(out, format="PNG")
    return out.getvalue()

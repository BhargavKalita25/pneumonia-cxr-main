import io
from fastapi.testclient import TestClient
from app.service.api import app
from PIL import Image

client=TestClient(app)

def test_health():
    r=client.get("/health")
    assert r.status_code==200 and r.json()["status"]=="ok"

def test_predict_dummy(monkeypatch):
    # skip heavy model load by mocking Predictor
    from app.service import api
    class DummyPred:
        def predict_with_cam(self,img,out_path): return "Normal",0.99
    api.PREDICTORS.clear(); api.PREDICTORS[("densenet121","outputs/best.ckpt")]=DummyPred()
    img=io.BytesIO(); Image.new("RGB",(224,224)).save(img,format="PNG"); img.seek(0)
    r=client.post("/predict",files={"file":("x.png",img,"image/png")})
    assert r.status_code==200
    assert "label" in r.json()

import os
import sys
import time
import atexit
import signal
import platform
import subprocess
import requests
import streamlit as st

# ===============================
# Backend auto-start (helper)
# ===============================
BACKEND_PROC = None

def _kill_backend():
    """Ensure the spawned backend process is terminated when Streamlit stops."""
    global BACKEND_PROC
    if BACKEND_PROC is not None and BACKEND_PROC.poll() is None:
        try:
            if platform.system() == "Windows":
                # Graceful terminate on Windows
                BACKEND_PROC.terminate()
            else:
                # Send SIGTERM on POSIX
                os.killpg(os.getpgid(BACKEND_PROC.pid), signal.SIGTERM)
        except Exception:
            pass
        BACKEND_PROC = None

@st.cache_resource(show_spinner=False)
def ensure_backend(api_base: str = "http://localhost:8000", autostart: bool = True, boot_timeout: float = 15.0):
    """
    Ensure a FastAPI backend is reachable at api_base.
    If not reachable and autostart=True, spawn a uvicorn server in the background and wait until /health is OK.

    Simple: Tries GET /health; if fails and autostart, starts uvicorn and retries until healthy or timeout.
    Technical: Spawns `python -m uvicorn app.service.api:app --host 0.0.0.0 --port 8000` via subprocess.
               Registers atexit cleanup to terminate the process. Uses cached resource to run only once per session.
    """
    global BACKEND_PROC

    health_url = f"{api_base}/health"

    def _healthy(timeout=2):
        try:
            r = requests.get(health_url, timeout=timeout)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    # If already up, nothing to do
    if _healthy():
        return True

    if not autostart:
        return False

    # Start backend (uvicorn) if not up
    st.sidebar.info("Starting backend…")
    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn",
        "app.service.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    # On POSIX, start a new process group for clean termination
    popen_kwargs = {}
    if platform.system() != "Windows":
        popen_kwargs.update(dict(preexec_fn=os.setsid))

    try:
        BACKEND_PROC = subprocess.Popen(uvicorn_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_kwargs)
        atexit.register(_kill_backend)
    except Exception as e:
        st.sidebar.error(f"Failed to launch backend: {e}")
        return False

    # Wait for health to be OK
    start = time.time()
    while time.time() - start < boot_timeout:
        if _healthy(timeout=1):
            st.sidebar.success("Backend started ✅")
            return True
        time.sleep(0.5)

    st.sidebar.error("Backend did not become healthy in time.")
    return False


# ===============================
# UI config
# ===============================
st.set_page_config(page_title="Pneumonia CXR Detection", layout="wide")
st.sidebar.title("Pneumonia Detection (Research/Education)")
st.sidebar.caption("⚠️ Not for diagnostic use")

# Decide API base: if API_URL is provided, prefer it; otherwise start/assume local backend.
ENV_API = os.environ.get("API_URL", "").strip()
API = ENV_API if ENV_API else "http://localhost:8000"

# If no external API_URL provided, ensure local backend is running
if not ENV_API:
    ok = ensure_backend(API, autostart=True, boot_timeout=20.0)
    if not ok:
        st.sidebar.error("Backend unavailable and could not be auto-started.")
else:
    # External API provided; check health but do not auto-start.
    try:
        r = requests.get(f"{API}/health", timeout=3)
        if r.status_code == 200:
            st.sidebar.success("Connected to external backend ✅")
        else:
            st.sidebar.warning(f"Backend health check: HTTP {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"Could not reach external backend: {e}")

# ===============================
# Fetch model info safely
# ===============================
try:
    model_meta = requests.get(f"{API}/models", timeout=5).json()[0]
    st.sidebar.write(f"Model: {model_meta['name']} | Input: {model_meta['input_size']}")
except Exception as e:
    st.sidebar.error(f"Could not connect to API /models: {e}")

# ===============================
# Main UI
# ===============================
st.title("Chest X-Ray Pneumonia Detection")

uploader = st.file_uploader(
    "Upload images (.png/.jpg/.jpeg, optional .dcm)",
    accept_multiple_files=True
)

def call_api_single(file):
    """Call /predict for a single image. Returns (json_or_none, err_or_none)."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        res = requests.post(
            f"{API}/predict",
            files=files,
            params={"enable_dicom": file.name.lower().endswith(".dcm")},
            timeout=30
        )
        if res.status_code == 200:
            return res.json(), None
        else:
            return None, f"API error {res.status_code}: {res.text}"
    except Exception as e:
        return None, f"Request failed: {e}"

def call_api_batch(files_list):
    """Call /predict-batch for multiple images. Returns (json_or_none, err_or_none)."""
    try:
        files = [("files", (f.name, f.getvalue(), f.type)) for f in files_list]
        res = requests.post(
            f"{API}/predict-batch",
            files=files,
            params={"enable_dicom": any(f.name.lower().endswith(".dcm") for f in files_list)},
            timeout=60
        )
        if res.status_code == 200:
            return res.json(), None
        else:
            return None, f"API error {res.status_code}: {res.text}"
    except Exception as e:
        return None, f"Request failed: {e}"

# Trigger predictions
if st.button("Predict", disabled=not uploader):
    if len(uploader) == 1:
        result, err = call_api_single(uploader[0])
        if err:
            st.error(err)
        else:
            st.subheader(f"Prediction: {result['label']} ({result['confidence']:.2%})")
            st.image(f"{API}{result['gradcam_url']}", caption="Grad-CAM Overlay")
    elif len(uploader) > 1:
        results, err = call_api_batch(uploader)
        if err:
            st.error(err)
        else:
            for item in results:
                if item["label"] == "Error":
                    st.error(f"{item['filename']} → Error: {item['gradcam_url']}")
                else:
                    st.write(f"**{item['filename']}** → {item['label']} ({item['confidence']:.2%})")
                    st.image(f"{API}{item['gradcam_url']}", use_container_width=True)

# ===============================
# Metrics dashboard
# ===============================
st.markdown("---")
st.subheader("Metrics Dashboard")

metrics_imgs = ["loss.png", "acc.png", "roc.png", "pr.png", "calibration.png"]
for img in metrics_imgs:
    p = os.path.join("outputs", img)
    if os.path.exists(p):
        st.image(p, caption=img)
    else:
        st.info(f"Metrics file not found: {p}")

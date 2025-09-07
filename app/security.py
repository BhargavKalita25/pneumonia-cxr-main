import time, os
from fastapi import HTTPException, UploadFile

RATE = {}
WINDOW = 10
ALLOW = 20

def rate_limiter(key: str):
    now = time.time()
    bucket = [t for t in RATE.get(key, []) if now - t < WINDOW]
    if len(bucket) >= ALLOW:
        raise HTTPException(status_code=429, detail="Too many requests")
    bucket.append(now)
    RATE[key] = bucket

def validate_upload(f: UploadFile, allowed_ext=(".png",".jpg",".jpeg",".dcm")):
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
    return True

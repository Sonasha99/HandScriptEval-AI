import os
import sys
import time
import urllib.request
import hashlib
from pathlib import Path

SNAPSHOT_DIR = Path(r"C:\Users\Asus\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3")
BLOBS_DIR = Path(r"C:\Users\Asus\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\blobs")

FILES = [
    {
        "filename": "model-00001-of-00002.safetensors",
        "url": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00001-of-00002.safetensors",
        "total_size": 3982649232,
        "sha256": "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8",
        "existing_incomplete": BLOBS_DIR / "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8.4dbaf706.incomplete"
    },
    {
        "filename": "model-00002-of-00002.safetensors",
        "url": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00002-of-00002.safetensors",
        "total_size": 3526688744,
        "sha256": "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4",
        "existing_incomplete": BLOBS_DIR / "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4.ae34a893.incomplete"
    }
]

def download_file(item):
    final_dest = SNAPSHOT_DIR / item["filename"]
    if final_dest.exists() and final_dest.stat().st_size == item["total_size"]:
        print(f"[{item['filename']}] Already complete in snapshot folder!")
        return

    work_file = item["existing_incomplete"]
    current_size = work_file.stat().st_size if work_file.exists() else 0
    total_size = item["total_size"]

    print(f"[{item['filename']}] Current size: {current_size / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB ({current_size/total_size*100:.1f}%)")

    chunk_size = 4 * 1024 * 1024  # 4MB chunks
    t_last = time.time()
    bytes_last = current_size

    while current_size < total_size:
        end_byte = min(current_size + chunk_size - 1, total_size - 1)
        req = urllib.request.Request(
            item["url"],
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={current_size}-{end_byte}"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if not data:
                    print("No data received, waiting 3s...")
                    time.sleep(3)
                    continue
                with open(work_file, "ab") as f:
                    f.write(data)
                current_size += len(data)
        except Exception as e:
            print(f"Error at byte {current_size}: {e}. Retrying in 2s...")
            time.sleep(2)
            continue

        now = time.time()
        if now - t_last >= 10:
            speed = ((current_size - bytes_last) / (1024**2)) / (now - t_last)
            eta_sec = (total_size - current_size) / (max(speed, 0.01) * 1024 * 1024)
            print(f"[{item['filename']}] {current_size/(1024**2):.1f}/{total_size/(1024**2):.1f} MB ({current_size/total_size*100:.1f}%) | {speed:.2f} MB/s | ETA: {eta_sec/60:.1f} min")
            t_last = now
            bytes_last = current_size

    print(f"[{item['filename']}] Download finished! Copying to snapshot directory...")
    # Copy/move to final destination in snapshot
    with open(work_file, "rb") as src, open(final_dest, "wb") as dst:
        while True:
            buf = src.read(16 * 1024 * 1024)
            if not buf:
                break
            dst.write(buf)
    print(f"[{item['filename']}] Successfully saved to {final_dest}!")

if __name__ == "__main__":
    for item in FILES:
        download_file(item)
    print("ALL QWEN FILES SUCCESSFULLY DOWNLOADED AND READY!")

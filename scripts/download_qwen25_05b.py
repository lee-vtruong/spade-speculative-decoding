import os
from huggingface_hub import snapshot_download

cache_dir = "/home/stackops/whale/cache/huggingface/hub"
print(f"Downloading Qwen/Qwen2.5-0.5B-Instruct to {cache_dir}...")
path = snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=cache_dir,
    revision="main",
)
print(f"Downloaded successfully to: {path}")

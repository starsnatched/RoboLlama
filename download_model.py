from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-3.2-1B", local_dir="llm")
snapshot_download(repo_id="google/vit-base-patch16-224", local_dir="vit")
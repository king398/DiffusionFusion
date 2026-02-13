import os 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

snapshot_download(repo_id="ILSVRC/imagenet-1k",local_dir="/work/nvme/betw/msalunkhe/data/imagenet",repo_type="dataset")

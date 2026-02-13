from datasets import load_dataset

ds = load_dataset("/work/nvme/betw/msalunkhe/data/imagenet",split="train")

print(ds[0])
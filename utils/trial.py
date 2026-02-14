from datasets import load_dataset
import os
from torchvision import transforms
from JiT.util.crop import center_crop_arr
import torch

os.environ["HF_HOME"] = "/work/nvme/betw/msalunkhe/data/huggingface"

transform_train = transforms.Compose([
    transforms.Lambda(lambda img: center_crop_arr(img, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.PILToTensor(),
])

def transform(examples):
    # examples is a dict of lists in batched mode
    examples["  "] = [
        transform_train(img.convert("RGB")) for img in examples["image"]
    ]
    return examples

def collate_fn(batch):
    # batch is list of dicts like {"image": tensor, "label": int, ...}
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b.get("label", -1) for b in batch], dtype=torch.long)
    return {"image": images, "label": labels}

def main():
    train_dir = "/work/nvme/betw/msalunkhe/data/imagenet/"

    ds = load_dataset(
        train_dir,
        split="train",
       ## streaming=True,
    )

    # Streaming: do sharding for DDP instead of DistributedSampler
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    # Shuffle with a buffer (bigger buffer = better shuffle, more RAM)
    ds = ds.shuffle(buffer_size=50_000, seed=0)
    ds = ds.shard(num_shards=world_size, index=rank)

    # Apply your torchvision transform on-the-fly
    ds = ds.with_format("torch")  # ensure PIL objects come through
    ds.set_transform(transform)

    data_loader_train = torch.utils.data.DataLoader(
        ds,
        batch_size=args.,
        num_workers=4,          # for IterableDataset, workers can help, but tune it
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    batch = next(iter(data_loader_train))
    print(batch["image"].shape, batch["label"].shape)

if __name__ == "__main__":
    main()
ds = load_dataset("timm/imagenet-1k-wds")

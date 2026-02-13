from datasets import load_dataset
import os
from torchvision import transforms
from JiT.util.crop import center_crop_arr
import torch
os.environ['HF_HOME'] = "/work/nvme/betw/msalunkhe/data/huggingface"


def transform(examples):
    examples["image"] = [transform_train(
        image.convert("RGB")) for image in examples["image"]]
    return


transform_train = transforms.Compose([
    transforms.Lambda(lambda img: center_crop_arr(img, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.PILToTensor()
])
ds = load_dataset("/work/nvme/betw/msalunkhe/data/imagenet")['train']
ds.set_transform(transform)
sampler_train = torch.utils.data.DistributedSampler(
    ds, num_replicas=1, rank=0, shuffle=True
)
print("Sampler_train =", sampler_train)
data_loader_train = torch.utils.data.DataLoader(
    ds, sampler=sampler_train,
    drop_last=True
)
print(next(iter(data_loader_train)))

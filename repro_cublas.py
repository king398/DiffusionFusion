"""Minimal repro of the Test B crash: bf16 nn.Linear on the dino embedder layout."""
import os, ctypes, torch, torch.nn as nn

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH", "")[:300])
# which libcublas is actually loaded into the process
try:
    import torch  # ensure cuda libs loaded
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda") @ torch.zeros(1, 1, device="cuda")
except Exception:
    pass
os.system("grep -i cublas /proc/self/maps | awk '{print $6}' | sort -u")

dev = torch.device("cuda")
lin = nn.Linear(768, 896).to(dev)

def trial(tag, x):
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = lin(x)
        torch.cuda.synchronize()
        print(f"[{tag}] OK  out={tuple(y.shape)} {y.dtype} contig_in={x.is_contiguous()}")
    except Exception as e:
        print(f"[{tag}] FAIL contig_in={x.is_contiguous()}: {type(e).__name__}: {str(e)[:160]}")

# exact dino layout: [B,768,16,16] -> flatten(2) -> [B,768,256] -> transpose -> [B,256,768] (non-contiguous)
B = 64
dino = torch.randn(B, 768, 16, 16, device=dev)
x_nc = dino.flatten(2).transpose(1, 2)
trial("dino-noncontig", x_nc)
trial("dino-contiguous", x_nc.contiguous())
trial("plain-2d", torch.randn(B * 256, 768, device=dev))
trial("plain-3d-contig", torch.randn(B, 256, 768, device=dev))

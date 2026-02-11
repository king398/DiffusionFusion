import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm


def _load_sample(args):
    idx, sample_dir = args
    sample_path = f"{sample_dir}/{idx:06d}.png"
    with Image.open(sample_path) as sample_pil:
        return np.asarray(sample_pil, dtype=np.uint8)


def create_npz_from_sample_folder(sample_dir, num=50_000, num_workers=None):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    chunk_size = max(1, num // (num_workers * 8))
    jobs = ((i, sample_dir) for i in range(num))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        iterator = executor.map(_load_sample, jobs, chunksize=chunk_size)
        samples = list(
            tqdm(
                iterator,
                total=num,
                desc=f"Building .npz file from samples ({num_workers} workers)",
            )
        )

    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == "__main__":
    sample_dir = "/projects/betw/msalunkhe/DiT_generations/DiT-XL-2-pretrained-size-256-vae-ema-cfg-1.5-seed-0"
    create_npz_from_sample_folder(sample_dir)

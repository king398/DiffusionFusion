from typing import Dict

import numpy as np
import torch

from JiT.util.feature_shards import (
    DatasetShardSpan,
    FeatureShardStore,
    LogicalShardSpan,
    MultiStreamShardDataset,
    PairedRamLoadedShardDataset,
    inspect_feature_shards,
    load_feature_range_to_ram,
    maybe_append_split_suffix,
    resolve_feature_dir_name,
    resolve_feature_dataset_root,
)


class RamLoadedShardDataset(PairedRamLoadedShardDataset):
    """Legacy 2-stream training batch wrapper.

    Retained so reverted-or-experimental code can still import the old name.
    New code should use :class:`MultiStreamRamLoadedShardDataset`.
    """

    def _format_batch(self, rows: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "latent": torch.from_numpy(rows["latent"]),
            # DINO features are normalized once during extraction.
            # Repeating layer norm here silently changes float16 shards.
            "dino": torch.from_numpy(rows["dino"]),
            "y": torch.from_numpy(rows["y"]),
        }


class MultiStreamRamLoadedShardDataset(MultiStreamShardDataset):
    """N-stream training batch wrapper.

    Emits dict batches keyed by stream name plus a ``y`` label tensor; drops
    the ``sample_id`` column that the underlying dataset surfaces for
    bookkeeping so the engine doesn't move it to GPU.
    """

    def _format_batch(self, rows: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out = {name: torch.from_numpy(rows[name]) for name in self.stream_names}
        out["y"] = torch.from_numpy(rows["y"])
        return out


__all__ = [
    "DatasetShardSpan",
    "FeatureShardStore",
    "LogicalShardSpan",
    "MultiStreamRamLoadedShardDataset",
    "MultiStreamShardDataset",
    "RamLoadedShardDataset",
    "inspect_feature_shards",
    "load_feature_range_to_ram",
    "maybe_append_split_suffix",
    "resolve_feature_dir_name",
    "resolve_feature_dataset_root",
]

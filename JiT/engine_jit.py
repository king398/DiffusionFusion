import math
import sys
import os
import shutil
import time
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import torch
import numpy as np

import JiT.util.misc as misc
import JiT.util.lr_sched as lr_sched
from JiT.eval.diffusion_decoder import decode_with_decoder
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None
try:
    import torch_fidelity
except ImportError:
    torch_fidelity = None


class _DeviceBatchPrefetcher:
    """Move the next CPU batch to the training device on a side CUDA stream."""

    def __init__(self, data_loader, device, enabled: bool = True):
        self._iterator = iter(data_loader)
        self._use_cuda = (
            enabled
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        if self._use_cuda:
            device_index = (
                device.index if device.index is not None else torch.cuda.current_device()
            )
            self._device = torch.device("cuda", device_index)
        else:
            self._device = device
        self._stream = torch.cuda.Stream(device=self._device) if self._use_cuda else None
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="jit-cuda-prefetch")
            if self._use_cuda
            else None
        )
        self._next_batch = None
        self._next_future = None
        self._preload()

    def _move_batch(self, batch):
        non_blocking = self._use_cuda
        return (
            batch["latent"].to(self._device, non_blocking=non_blocking),
            batch["dino"].to(self._device, non_blocking=non_blocking),
            batch["y"].to(self._device, non_blocking=non_blocking).view(-1).long(),
        )

    def _load_next_batch(self):
        batch = next(self._iterator)
        if self._use_cuda:
            torch.cuda.set_device(self._device)
            with torch.cuda.stream(self._stream):
                return self._move_batch(batch)
        return self._move_batch(batch)

    def _preload(self):
        if self._executor is not None:
            self._next_future = self._executor.submit(self._load_next_batch)
            return
        try:
            self._next_batch = self._load_next_batch()
        except StopIteration:
            self._next_batch = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._executor is not None:
            if self._next_future is None:
                raise StopIteration
            try:
                batch = self._next_future.result()
            except StopIteration:
                self._next_future = None
                raise
            current_stream = torch.cuda.current_stream(self._device)
            current_stream.wait_stream(self._stream)
            for tensor in batch:
                tensor.record_stream(current_stream)
            self._preload()
            return batch

        if self._next_batch is None:
            raise StopIteration

        batch = self._next_batch
        self._preload()
        return batch

    def close(self):
        if self._next_future is not None:
            self._next_future.cancel()
            self._next_future = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


def _all_reduce_loss_tensor(loss: torch.Tensor) -> float:
    reduced = loss.detach()
    if misc.get_world_size() > 1:
        reduced = reduced.clone()
        torch.distributed.all_reduce(reduced)
        reduced /= misc.get_world_size()
    return float(reduced.item())


@contextmanager
def _swap_ema_parameters(model_without_ddp, ema_params):
    params = list(model_without_ddp.parameters())
    if len(params) != len(ema_params):
        raise ValueError(
            f"EMA parameter count mismatch: model has {len(params)} parameters, "
            f"EMA has {len(ema_params)}."
        )
    for param, ema_param in zip(params, ema_params):
        param.data, ema_param.data = ema_param.data, param.data
    try:
        yield
    finally:
        for param, ema_param in zip(params, ema_params):
            param.data, ema_param.data = ema_param.data, param.data


def _save_png(path: str, image: np.ndarray) -> None:
    Image.fromarray(image).save(path, format="PNG", compress_level=0)


def _drain_save_futures(futures, *, limit: int | None = None) -> None:
    while futures and (limit is None or len(futures) >= limit):
        done, _pending = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            future.result()
            futures.remove(future)


def train_one_epoch(
    model,
    model_without_ddp,
    data_loader,
    optimizer,
    device,
    epoch,
    log_writer=None,
    args=None,
    steps_per_epoch: int = None,
    optimizer_steps_per_epoch: int = None,
    wandb_run=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    log_freq = max(1, int(getattr(args, "log_freq", 1)))
    cuda_prefetch = bool(getattr(args, "cuda_prefetch", True))
    steps_per_epoch = steps_per_epoch or len(data_loader)
    optimizer_steps_per_epoch = optimizer_steps_per_epoch or math.ceil(
        steps_per_epoch / accum_iter
    )

    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    device_batches = _DeviceBatchPrefetcher(
        data_loader,
        device,
        enabled=cuda_prefetch,
    )
    consumed_micro_batches = 0

    try:
        for optimizer_step, _ in enumerate(
            metric_logger.log_every(
                range(optimizer_steps_per_epoch),
                print_freq,
                header,
                optimizer_steps_per_epoch,
            )
        ):
            # Per optimizer step (instead of per micro-batch) lr scheduler so accumulation
            # matches the reference large-batch JiT training recipe.
            lr = lr_sched.adjust_learning_rate(
                optimizer,
                optimizer_step / optimizer_steps_per_epoch + epoch,
                args,
            )
            remaining_micro_batches = steps_per_epoch - consumed_micro_batches
            micro_batches_in_update = min(accum_iter, remaining_micro_batches)
            if micro_batches_in_update <= 0:
                break
            accum_loss = torch.zeros((), device=device)

            for _micro_batch_idx in range(micro_batches_in_update):
                latent, dino, labels = next(device_batches)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(latent, dino, labels)

                accum_loss = accum_loss + loss.detach()
                (loss / micro_batches_in_update).backward()
                consumed_micro_batches += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model_without_ddp.update_ema()

            metric_logger.update(lr=lr)
            completed_optimizer_steps = optimizer_step + 1
            should_measure_loss = (
                optimizer_step % log_freq == 0
                or optimizer_step % print_freq == 0
                or completed_optimizer_steps == optimizer_steps_per_epoch
            )

            if should_measure_loss:
                step_loss_tensor = accum_loss / micro_batches_in_update
                loss_value_reduce = _all_reduce_loss_tensor(step_loss_tensor)
                if not math.isfinite(loss_value_reduce):
                    print("Loss is {}, stopping training".format(loss_value_reduce))
                    sys.exit(1)

                metric_logger.update(loss=loss_value_reduce)
            else:
                loss_value_reduce = None

            if log_writer is not None:
                # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
                epoch_1000x = int(
                    (epoch + completed_optimizer_steps / optimizer_steps_per_epoch) * 1000
                )
                if loss_value_reduce is not None and optimizer_step % log_freq == 0:
                    log_writer.add_scalar(
                        'train_loss', loss_value_reduce, epoch_1000x)
                    log_writer.add_scalar('lr', lr, epoch_1000x)
            if wandb_run is not None and loss_value_reduce is not None and optimizer_step % log_freq == 0:
                global_step = epoch * optimizer_steps_per_epoch + optimizer_step
                payload = {
                    "train/loss": loss_value_reduce,
                    "train/lr": lr,
                    "train/epoch_progress": epoch + completed_optimizer_steps / optimizer_steps_per_epoch,
                }
                misc.add_wandb_global_step(payload, global_step)
                wandb_run.log(payload)
    finally:
        device_batches.close()
    print(f"Finished ")


@torch.inference_mode()
def evaluate(model_without_ddp, args, epoch, decoder, batch_size=64, log_writer=None, wandb_run=None, wandb_step=None):
    print("Start evaluation at epoch {}".format(epoch))
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    global_batch_size = batch_size * world_size
    num_steps = math.ceil(args.num_images / global_batch_size)
    eval_image_interval = max(1, getattr(
        args, "wandb_eval_image_interval", 10))
    wandb_table = None
    if misc.is_main_process() and wandb_run is not None:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb` or disable --use_wandb."
            )
        wandb_table = wandb.Table(
            columns=["epoch", "global_index", "class_id", "image"]
        )

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.latent_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    misc.distributed_barrier()

    # switch to ema params, hard-coded to be the first one
    print("Switch to ema")
    class_num = args.class_num
    class_label_gen_world = np.arange(
        0, class_num, dtype=np.int64).repeat(math.ceil(args.num_images / class_num))[:args.num_images]

    save_workers = max(0, int(getattr(args, "eval_save_workers", 4)))
    max_pending_saves = max(1, save_workers * 8)
    save_executor = (
        ThreadPoolExecutor(
            max_workers=save_workers,
            thread_name_prefix=f"jit-eval-save-r{local_rank}",
        )
        if save_workers > 0
        else None
    )
    save_futures = []
    try:
        with _swap_ema_parameters(model_without_ddp, model_without_ddp.ema_params1):
            for step_idx in range(num_steps):
                print("Generation step {}/{}".format(step_idx, num_steps))

                start_idx = world_size * batch_size * step_idx + local_rank * batch_size
                end_idx = start_idx + batch_size
                labels_gen_np = class_label_gen_world[start_idx:min(end_idx, args.num_images)].copy()
                labels_gen = np.zeros(batch_size, dtype=np.int64)
                labels_gen[:labels_gen_np.shape[0]] = labels_gen_np
                labels_gen = torch.from_numpy(labels_gen).long().cuda()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    sampled_latents, sampled_dino = model_without_ddp.generate(
                        labels_gen)

                print("Decoding step {}/{}".format(step_idx, num_steps))
                sampled_images = decode_with_decoder(decoder, sampled_latents, sampled_dino)
                for sample_idx, sample in enumerate(sampled_images):
                    index = sample_idx + world_size * batch_size * \
                        step_idx + local_rank * batch_size
                    if index >= args.num_images:
                        continue

                    image_path = os.path.join(save_folder, '{}.png'.format(
                        str(index).zfill(5)))
                    if save_executor is None:
                        _save_png(image_path, sample)
                    else:
                        save_futures.append(
                            save_executor.submit(_save_png, image_path, sample.copy())
                        )
                        _drain_save_futures(save_futures, limit=max_pending_saves)
                    if wandb_table is not None and index % eval_image_interval == 0:
                        class_id = int(labels_gen_np[sample_idx])
                        wandb_table.add_data(
                            epoch,
                            index,
                            class_id,
                            wandb.Image(
                                sample, caption=f"class={class_id}, idx={index}")
                        )
    finally:
        _drain_save_futures(save_futures)
        if save_executor is not None:
            save_executor.shutdown(wait=True)

    misc.distributed_barrier()

    # back to no ema
    print("Switch back from ema")

    # compute FID and IS on rank 0, while other ranks wait outside NCCL collectives
    metrics_requested = bool(getattr(args, "output_dir", None)) or bool(
        getattr(args, "use_wandb", False)
    )
    metrics_done_path = os.path.join(
        args.output_dir if getattr(args, "output_dir", None) else ".",
        f".eval_metrics_done_epoch_{int(epoch):05d}",
    )
    if metrics_requested and misc.is_main_process() and os.path.exists(metrics_done_path):
        os.remove(metrics_done_path)
    if metrics_requested:
        misc.distributed_barrier()

    if metrics_requested and misc.is_main_process():
        if torch_fidelity is None:
            raise ImportError("torch_fidelity is required for JiT generation metrics.")
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=args.fid_stats_path,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(
            model_without_ddp.cfg_scale, args.latent_size)
        if log_writer is not None:
            log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
            log_writer.add_scalar('is{}'.format(postfix),
                                  inception_score, epoch)
        if misc.is_main_process() and wandb_run is not None:
            log_payload = {
                'eval/fid{}'.format(postfix): fid,
                'eval/is{}'.format(postfix): inception_score,
            }
            if wandb_table is not None and len(wandb_table.data) > 0:
                log_payload['eval/samples{}'.format(postfix)] = wandb_table
            misc.add_wandb_global_step(log_payload, wandb_step)
            wandb_run.log(log_payload)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(
            fid, inception_score))
        with open(metrics_done_path, "w", encoding="utf-8") as f:
            f.write(f"{fid},{inception_score}\n")
    elif metrics_requested:
        wait_start = time.time()
        wait_timeout = float(getattr(args, "dist_timeout_sec", 7200))
        while not os.path.exists(metrics_done_path):
            if time.time() - wait_start > wait_timeout:
                raise RuntimeError(
                    f"Timed out waiting for eval metrics sync file: {metrics_done_path}"
                )
            time.sleep(1.0)

    misc.distributed_barrier()
    if misc.is_main_process() and metrics_requested:
        if os.path.isdir(save_folder):
            shutil.rmtree(save_folder)
        if os.path.exists(metrics_done_path):
            os.remove(metrics_done_path)

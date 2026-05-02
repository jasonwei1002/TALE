import argparse
import dataclasses
import logging.config
import pprint
import random
from contextlib import nullcontext
from os import path, makedirs
from time import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab
import configs
from data import transforms, utils as datautils
from data.datasets import (
  DATASETS,
  CODE15,
  StPetersburg,
  PTB_XL
)
from data.masks import MaskCollator
from data.datasets.finetune_dataset import getdataset as get_finetune_dataset
from data.utils import (
  TensorDataset,
  LazyTensorDataset,
  VariableTensorDataset,
  DatasetRouter
)
from models import JEPA
from utils.monitoring import (
  AverageMeter,
  get_cpu_count,
  get_memory_usage
)
from utils.schedules import (
  linear_schedule,
  cosine_schedule,
  update_weight_decay_,
  update_learning_rate_
)

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+', required=True, help='list of dataset=path/to/data.npy pairs')
parser.add_argument('--out', default='pretrain', help='output directory')
parser.add_argument('--config', default='ViTS_mimic', help='path to config file or config name')
parser.add_argument('--chkpt', help='resume training from model checkpoint')
parser.add_argument('--amp', default='bfloat16', choices=['bfloat16', 'float32'], help='automated mixed precision')
parser.add_argument('--compile', action='store_true', help='compile model')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# linear probing arguments
parser.add_argument('--probe-data', help='path to PTB-XL data directory for linear probing')
parser.add_argument('--probe-train-csv', help='path to train CSV file for linear probing')
parser.add_argument('--probe-val-csv', help='path to validation CSV file for linear probing')
parser.add_argument('--probe-interval', type=int, default=10000, help='interval for linear probing evaluation (0 to disable)')
parser.add_argument('--probe-steps', type=int, default=1000, help='training steps for linear probe')
args = parser.parse_args()

# NOTE: we compute mean and standard deviation of ptb-xl over the train folds (1-8).
#  We only use these folds during pre-training.
PTB_XL.mean = [-0.002, -0.002, 0.000, 0.002, -0.001, -0.001,
               0.000, -0.001, -0.002, -0.001, -0.001, -0.001]
PTB_XL.std = [0.191, 0.166, 0.173, 0.142, 0.149, 0.147,
              0.235, 0.338, 0.335, 0.299, 0.294, 0.242]


def seed_everything(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def make_generator(seed):
  generator = torch.Generator()
  generator.manual_seed(seed)
  return generator


def main():
  makedirs(args.out, exist_ok=True)
  logging.config.fileConfig('logging.ini')
  logger = logging.getLogger('app')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  using_cuda = device.type == 'cuda'
  num_cpus = get_cpu_count()
  logger.debug(f'using {device} accelerator and {num_cpus} CPUs')

  if using_cuda:
    logger.debug('TF32 tensor cores are enabled')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
  seed_everything(args.seed)

  if args.amp == 'float32' or not using_cuda:  # don't use AMP on a CPU
    logger.debug('using float32 precision')
    auto_mixed_precision = nullcontext()
  elif args.amp == 'bfloat16':
    # bfloat16 preserves the range of float32, so it does not require scaling
    logger.debug('using bfloat16 with AMP')
    auto_mixed_precision = torch.cuda.amp.autocast(dtype=torch.bfloat16)
  else:
    raise ValueError('Failed to choose floating-point format.')

  if args.chkpt:
    logger.debug(f'resuming from checkpoint {args.chkpt}')
    chkpt = torch.load(args.chkpt, map_location=device)
    config = configs.pretrain.Config(**chkpt['config'])
  else:
    # read config file
    if not path.isfile(args.config):
      # maybe config is the name of a default config file in configs/pretrain/
      config_file = path.join(path.dirname(configs.pretrain.__file__),  f'{args.config}.yaml')
      if not path.isfile(config_file):
        raise ValueError(f'Failed to read configuration file {args.config}')
      args.config = config_file
    config_dict = configs.load_config_file(args.config)
    config = configs.pretrain.Config(**config_dict)
    logger.debug(f'loading configuration file from {args.config}:\n'
                 f'{pprint.pformat(config_dict, compact=True, sort_dicts=False, width=120)}')
    chkpt = None

  if getattr(config, 'encoder_type', 'vit1d') != 'vit1d':
    raise ValueError(f'Unsupported encoder_type: {config.encoder_type}')
  
  run = swanlab.init(
    # 设置项目
    project="ecg-jepa",
    # 跟踪超参数与实验元数据
    config={
        "learning_rate": config.learning_rate,
        "epochs": config.steps,
        "seed": args.seed,
    },
  )

  dump_files = {}
  for data_arg in args.data:
    dataset_name, *maybe_dump_file = data_arg.split('=', 1)
    if not maybe_dump_file:
      raise ValueError('Dataset pair must have following format: dataset=path/to/data.npy')
    dump_file, = maybe_dump_file
    dump_files[dataset_name] = dump_file

  for dataset_name in config.datasets:
    if dataset_name not in DATASETS:
      raise ValueError(f'Unknown dataset {dataset_name}. '
                       f'Available datasets are {list(DATASETS)}')
    if dataset_name not in dump_files:
      raise ValueError(f'Missing {dataset_name} dataset in `--data` argument')
    dump_file = dump_files[dataset_name]
    if not path.isfile(dump_file):
      raise ValueError(f'Dataset does not exist {dump_file}')
    _, ext = path.splitext(dump_file)
    if ext not in ('.npy', '.npz'):
      raise ValueError(f'Unsupported dataset format: {dump_file}')

  datasets = {}
  for dataset_name, weight in config.datasets.items():
    dump_file = dump_files[dataset_name]
    logger.debug(f'loading {dataset_name} from {dump_file}')
    dataset_cls = DATASETS[dataset_name]
    resample_ratio = config.sampling_frequency / dataset_cls.sampling_frequency
    channel_order = datautils.get_channel_order(dataset_cls.channels, config.channels)
    mean = np.array([dataset_cls.mean], dtype=np.float16)
    std = np.array([dataset_cls.std], dtype=np.float16)
    _, ext = path.splitext(dump_file)
    if ext == '.npy':
      dataset = LazyTensorDataset(
        dump_file=dump_file,
        preprocess=PreprocessECG(
          mean_std=(mean, std),
          resample_ratio=resample_ratio,
          channel_order=channel_order),
        transform=TransformECG(
          crop_size=config.channel_size))
    elif ext == '.npz':
      dataset = VariableTensorDataset(
        *load_variable_data_dump(
          dump_file=dump_file,
          min_channel_size=config.channel_size,
          transform=PreprocessECG(
            mean_std=(mean, std),
            resample_ratio=resample_ratio,
            channel_order=channel_order),
          processes=min(num_cpus, 12)),  # Limit workers to avoid memory issues
        transform=TransformECG(
          crop_size=config.channel_size))
    else:
      raise ValueError(f'Unsupported dataset format: {dump_file}')
    datasets[dataset_name] = (dataset, weight)

  logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

  train_loader = DataLoader(
    dataset=DatasetRouter(datasets.values()),
    batch_size=config.batch_size,
    pin_memory=using_cuda,
    collate_fn=MaskCollator(
      patch_size=config.patch_size,
      min_block_size=config.min_block_size,
      min_keep_ratio=config.min_keep_ratio,
      max_keep_ratio=config.max_keep_ratio,
      num_patches=config.num_patches),
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=make_generator(args.seed))

  def map_to_device(data_iterator, device=None):
    for batch in data_iterator:
      yield tuple(x.to(device, non_blocking=using_cuda) for x in batch)

  def prefetch_batch(data_iterator):
    prefetched_batch = next(data_iterator)
    for next_batch in data_iterator:
      yield prefetched_batch
      prefetched_batch = next_batch
    yield prefetched_batch

  # if device is CUDA, batch data will be asynchronously transferred to the GPU,
  #  so we should perform as many CPU operations as possible between loading and using a batch
  train_iterator = iter(train_loader)
  train_iterator = map_to_device(train_iterator, device=device)
  train_iterator = prefetch_batch(train_iterator)

  # setup hyperparameter schedules
  if chkpt is not None:
    step = chkpt['step']
  else:
    step = 0

  momentum_schedule = linear_schedule(
    total_steps=config.steps,
    start_value=config.encoder_momentum,
    final_value=config.final_encoder_momentum,
    step=step)
  lr_schedule = cosine_schedule(
    total_steps=config.steps,
    start_value=config.learning_rate,
    final_value=config.final_learning_rate,
    warmup_steps=config.learning_rate_warmup_steps,
    warmup_start_value=1e-6,
    step=step)
  wd_schedule = cosine_schedule(
    total_steps=config.steps,
    start_value=config.weight_decay,
    final_value=config.final_weight_decay,
    step=step)

  # setup model
  model = original_model = JEPA(
    config=config,
    momentum_schedule=momentum_schedule,
    use_sdp_kernel=using_cuda
  ).to(device)
  optimizer = model.get_optimizer(fused=using_cuda)

  if chkpt is not None:  # resume training from checkpoint
    model.load_state_dict(chkpt['model'])
    optimizer.load_state_dict(chkpt['optimizer'])

  if args.compile:
    model = torch.compile(model)

  # Setup linear probing if enabled
  probe_train_loader = None
  probe_val_loader = None
  probe_num_classes = None
  best_probe_auc = float('-inf')
  best_probe_step = None
  
  if args.probe_data and args.probe_train_csv and args.probe_val_csv and args.probe_interval > 0:
    logger.debug(f'setting up linear probing with PTB-XL dataset')
    
    # Load probe datasets using the custom ECGDataset
    probe_train_dataset = get_finetune_dataset(
      data_path=args.probe_data,
      csv_path=args.probe_train_csv,
      mode='train',
      dataset_name='ptbxl')
    
    probe_val_dataset = get_finetune_dataset(
      data_path=args.probe_data,
      csv_path=args.probe_val_csv,
      mode='val',
      dataset_name='ptbxl')
    
    probe_num_classes = probe_train_dataset.num_classes
    
    probe_train_loader = DataLoader(
      dataset=probe_train_dataset,
      batch_size=128,
      shuffle=True,
      drop_last=True,
      num_workers=8,
      worker_init_fn=seed_worker,
      generator=make_generator(args.seed + 1))
    
    probe_val_loader = DataLoader(
      dataset=probe_val_dataset,
      batch_size=128,
      num_workers=8,
      worker_init_fn=seed_worker,
      generator=make_generator(args.seed + 2))
    
    logger.debug(f'linear probing setup complete: {probe_num_classes} classes, '
                 f'{len(probe_train_dataset)} train samples, {len(probe_val_dataset)} val samples')

  step_time = AverageMeter()
  train_loss = AverageMeter()

  pbar = tqdm(range(config.steps), desc='Pretraining', dynamic_ncols=True)
  for step in pbar:
    step_start = time()
    # update hyperparameters according to schedule
    update_learning_rate_(optimizer, next(lr_schedule))
    update_weight_decay_(optimizer, next(wd_schedule))
    # forward and backward pass
    batch_loss = 0.
    for _ in range(config.gradient_accumulation_steps):
      x, mask_encoder, mask_predictor = next(train_iterator)
      with auto_mixed_precision:
        loss = model(x, mask_encoder, mask_predictor)
        loss = loss / config.gradient_accumulation_steps
      loss.backward()
      batch_loss += loss.item()
    
    swanlab.log({
      'train/loss': batch_loss,
      'train/lr': optimizer.param_groups[0]['lr'],
    })
    # update weights
    if config.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    train_loss.update(batch_loss)
    optimizer.zero_grad(set_to_none=True)
    # finalize train step
    step_end = time()
    step_time.update(step_end - step_start)
    # Update progress bar
    pbar.set_postfix({'loss': f'{train_loss.value:.4f}'})
    if (step + 1) % 100 == 0:
      step_time = AverageMeter()
      train_loss = AverageMeter()
    if (step + 1) % config.checkpoint_interval == 0:
      torch.save({
        'model': original_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': dataclasses.asdict(config),
        'step': step + 1,
      }, path.join(args.out, f'chkpt_{step + 1}.pt'))
    
    # Linear probing evaluation
    if probe_train_loader is not None and (step + 1) % args.probe_interval == 0:
      probe_auc = run_linear_probe(
        target_encoder=original_model.target_encoder,
        probe_train_loader=probe_train_loader,
        probe_val_loader=probe_val_loader,
        num_classes=probe_num_classes,
        dim=config.dim,
        device=device,
        steps=args.probe_steps)
      swanlab.log({
        'probe/val_auc': probe_auc,
      })
      is_best = probe_auc > best_probe_auc
      if is_best:
        best_probe_auc = probe_auc
        best_probe_step = step + 1
        torch.save({
          'model': original_model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'config': dataclasses.asdict(config),
          'step': step + 1,
          'probe_auc': probe_auc,
        }, path.join(args.out, 'best_chkpt.pt'))
      logger.info(f'[{step + 1:06d}] {"(*)" if is_best else "   "} probe_auc {probe_auc:.4f}')


def load_variable_data_dump(dump_file, min_channel_size, transform=None, processes=None):
  data = datautils.load_variable_data_dump(dump_file, transform=transform, processes=processes)
  data = [x for x in data if len(x) >= min_channel_size]
  sizes = np.array([len(x) for x in data])
  starts = np.concatenate([np.array([0]), np.cumsum(sizes[:-1])])
  data = np.concatenate(data)
  return data, starts, sizes


class PreprocessECG:  # called once when loading the data
  def __init__(self, *, mean_std, resample_ratio, channel_order):
    self.mean, self.std = mean_std
    self.resample_ratio = resample_ratio
    self.channel_order = channel_order

  def __call__(self, x):
    # Transpose from (num_channels, channel_size) to (channel_size, num_channels)
    # User's data is stored as (12, 5000) but code expects (5000, 12)
    if x.shape[0] == 12 and x.shape[1] != 12:
      x = x.T
    # Ensure float32 dtype for normalization operations
    # Data was stored as int16 after multiplying by 1000, so divide to restore
    if x.dtype != np.float32:
      x = x.astype(np.float32) / 1000.0
    transforms.interpolate_NaNs_(x)
    if self.resample_ratio != 1.0:
      channel_size, num_channels = x.shape
      channel_size = int(self.resample_ratio * channel_size)
      x = transforms.resample(x, channel_size)
    # Skip normalization - data is already normalized
    # transforms.normalize_(x, mean_std=(self.mean, self.std))
    x.clip(-5, 5, out=x)
    x = x[:, self.channel_order]
    return x


class TransformECG:  # called whenever dataloader accesses the data
  def __init__(self, crop_size):
    self.crop_size = crop_size

  def __call__(self, x):
    x = transforms.random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


def run_linear_probe(
    target_encoder,
    probe_train_loader,
    probe_val_loader,
    num_classes,
    dim,
    device,
    steps=1000,
    learning_rate=1e-3
):
  """
  Run a fast linear probe to evaluate encoder representation quality.
  
  Args:
      target_encoder: The target encoder from JEPA model (will be frozen)
      probe_train_loader: DataLoader for training data
      probe_val_loader: DataLoader for validation data
      num_classes: Number of classes for classification
      dim: Encoder output dimension
      device: Compute device
      steps: Number of training steps for the probe
      learning_rate: Learning rate for probe training
      
  Returns:
      Validation macro AUC score
  """
  # Freeze encoder and set to eval mode
  target_encoder.eval()
  
  # Create a simple linear probe head
  probe_head = nn.Linear(dim, num_classes).to(device)
  optimizer = torch.optim.AdamW(probe_head.parameters(), lr=learning_rate)
  
  # Training loop helper
  def cycle(dataloader):
    while True:
      yield from dataloader
  
  train_iter = cycle(probe_train_loader)
  
  # Train the probe
  probe_head.train()
  for _ in tqdm(range(steps), desc='  Probe training', leave=False):
    x, y = next(train_iter)
    # Ensure tensors are float32 and on correct device
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(x)
    x = x.float().to(device)
    y = y.float().to(device)
    
    with torch.no_grad():
      features = target_encoder(x)
      features = features.mean(dim=1)  # global average pooling
    
    logits = probe_head(features)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  # Evaluate on validation set
  probe_head.eval()
  all_logits, all_targets = [], []
  with torch.no_grad():
    for x, y in tqdm(probe_val_loader, desc='  Probe validation', leave=False):
      # Ensure tensors are float32 and on correct device
      if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
      x = x.float().to(device)
      y = y.float().to(device)
      features = target_encoder(x)
      features = features.mean(dim=1)
      logits = probe_head(features)
      all_logits.append(logits.cpu())
      all_targets.append(y.cpu())
  
  predictions = torch.cat(all_logits).sigmoid().numpy()
  targets = torch.cat(all_targets).numpy()
  val_auc = roc_auc_score(targets, predictions, average='macro')
  # Restore encoder to train mode
  target_encoder.train()
  
  return val_auc


if __name__ == '__main__':
  main()

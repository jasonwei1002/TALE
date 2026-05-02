import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
import os
from torch import optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import sys
import pytz
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from config import config
from utils_trainer import Trainer
from utils_dataset import MIMIC_Dataset
import utils_builder

import swanlab
import warnings

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def ddp_main():
    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()
    #UTC+8 time
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(shanghai_tz).strftime("%Y%m%d_%H%M%S")
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    # set up

    if device_id == 0:
        swanlab.init(
            # Set the project where this run will be logged
            project="MERL_pretrain",
            name = config['swanlab_name'] + '_' + current_time,
            # Track hyperparameters and run metadata
            config={
                    "learning_rate": config['optimizer']['params']['lr'],
                    "total_epochs": config['trainer']['max_epochs'],
                    'weight_decay': config['optimizer']['params']['weight_decay'],
                    'ecg_model': config['network']['ecg_model'],
                    'text_model': config['network']['text_model'],
                    'batch_size': config['trainer']['batch_size'],
                    'val_zeroshot': 'all_sets',
                    'prompt_type': config['zeroshot']['prompt_type'],
            }
        )

    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)

    # define image-text dataset
    data_path = config['dataset']['data_path']
    dataset = MIMIC_Dataset(
        data_path=data_path, dataset_name=config['dataset']['dataset_name'])
    train_dataset = dataset.get_dataset(train_test='train')
    val_dataset = dataset.get_dataset(train_test='val')

    # building model part
    # --------------------
    model = utils_builder.ECGCLIP(config['network'])
    
    '''
    you can freeze bert from last layer to first layer.
    set num of layer in config.yaml
    default is freeze 9 layers
    '''
    if config['network']['free_layers'] is not None:
        for layer_idx in range(int(config['network']['free_layers'])):
            for param in list(model.lm_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # --------------------

    # --------------------
    opt_params = dict(config['optimizer']['params'])
    lr_text = opt_params.pop('lr_text', None)
    if lr_text is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            **opt_params,
        )
    else:
        # Separate LR for text-side modules (BERT + text projector) to reduce embedding space drift.
        text_params = list(model.module.lm_model.parameters()) + list(model.module.proj_t.parameters())
        text_param_ids = {id(p) for p in text_params}
        other_params = [p for p in model.module.parameters() if id(p) not in text_param_ids]

        text_params = [p for p in text_params if p.requires_grad]
        other_params = [p for p in other_params if p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params},
                {"params": text_params, "lr": lr_text},
            ],
            **opt_params,
        )


    trainer = Trainer(model=model,
                            optimizer=optimizer,
                            device=rank,
                            model_name=config['swanlab_name'],
                            **config['trainer'])
    # --------------------
    
    # --------------------

    trainer.fit(train_dataset, val_dataset, config['zeroshot'])




if __name__ == '__main__':
    ddp_main()

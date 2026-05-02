#!/bin/bash

python -m pretrain \
  --data "mimic-iv-ecg=/public/home/hs_mmcd_5/project/jasonwei/datasets/pretrain/train.npy" \
  --probe-data "/public/home/hs_mmcd_5/project/jasonwei/datasets/finetune/ptbxl" \
  --probe-train-csv "/public/home/hs_mmcd_5/project/jasonwei/datasets/finetune/data_split/ptbxl/super_class/ptbxl_super_class_train.csv" \
  --probe-val-csv "/public/home/hs_mmcd_5/project/jasonwei/datasets/finetune/data_split/ptbxl/super_class/ptbxl_super_class_val.csv" \
  --probe-interval 2500 \
  --probe-steps 5000 \
  --out "pretrain-with-probe" \
  --config "ViTS_mimic"

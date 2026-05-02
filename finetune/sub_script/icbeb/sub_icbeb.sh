task_name=$1
backbone=$2
pretrain_path=$3
ckpt_dir="/public/home/hs_mmcd_5/project/jasonwei/MERL/checkpoints/finetune/ckpt/icbeb/$task_name"
cd /public/home/hs_mmcd_5/project/jasonwei/MERL/finetune/

python main_single.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset icbeb \
    --pretrain_path $pretrain_path \
    --ratio 1 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 1000 \
    --name $task_name

python main_single.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset icbeb \
    --pretrain_path $pretrain_path \
    --ratio 10 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 1000 \
    --name $task_name

python main_single.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset icbeb \
    --pretrain_path $pretrain_path \
    --ratio 100 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 1000 \
    --name $task_name
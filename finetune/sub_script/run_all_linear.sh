taskname='linear'
backbone='vit_small'
pretrain_path='/public/home/hs_mmcd_5/project/jasonwei/MERL/checkpoints/best/resnet18_bestZeroShotAll_encoder.pth'

cd icbeb
bash sub_icbeb.sh $taskname $backbone $pretrain_path

cd ..
cd chapman
bash sub_chapman.sh $taskname $backbone $pretrain_path

cd ..
cd ptbxl
bash sub_ptbxl.sh $taskname $backbone $pretrain_path

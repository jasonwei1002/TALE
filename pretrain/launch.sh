export OMP_NUM_THREADS=5

torchrun --nproc_per_node=4 main.py
# pytorch-Multi-GPU

需要把data下面raw和pro进行解压。然后运行
‘’‘
python -m torch.distributed.launch --nproc_per_node 2 --use_env main.py
’‘’

from datetime import datetime
import torch
from pathlib import Path
# 18 layers, freeze, connection 0-6, No MLP, eval every 1000 iters, eval 500 iters, batch size 12, gradient accumulation steps 40, max iters 20000, learning rate 3e-6, decay lr, warmup iters 1000, lr decay iters 19000, min lr 3e-7
name = Path(__file__).stem
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H")
out_name = f"{timestamp}_connection_list_0_6"
n_layer= 18
out_dir = out_name
eval_interval = 1000
eval_iters = 500
wandb_log = True # feel free to turn on
wandb_project = 'openwebtext-configs'
wandb_run_name = out_name # 'Connection list' is the name of the experiment, you can change it to something else
init_from = 'gpt2' # this is the largest GPT-2 model #original
always_save_checkpoint = True
batch_size = 12 
gradient_accumulation_steps = 40
max_iters = 5000
device = 'cuda'
learning_rate = 3e-6
decay_lr = True
freeze = False
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = max_iters - 100 # should be ~= max_iters per Chinchilla
min_lr = 3e-7 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
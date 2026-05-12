from datetime import datetime
import torch
from pathlib import Path

name = Path(__file__).stem
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H")
out_name = f"{timestamp}_finetune20k"
n_layer= 12
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
max_iters = 20000
device = 'cuda'
learning_rate = 3e-7 #original
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = max_iters - 100 # should be ~= max_iters per Chinchilla
min_lr = 0.1 * learning_rate # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

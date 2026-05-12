from datetime import datetime
import torch
from pathlib import Path

name = Path(__file__).stem
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H")
# model with new layer with connection (trained with freezing/ random init)
n_layer= 13
out_dir =  timestamp + '/' + 'out-openwebtext-configs' 
eval_interval = 1000
eval_iters = 500
wandb_log = True # feel free to turn on
wandb_project = 'openwebtext-configs'
wandb_run_name = 'Grow-13-freeze-connection-random-init'
init_from = 'gpt2' # this is the largest GPT-2 model #original
always_save_checkpoint = True
batch_size = 12
gradient_accumulation_steps = 40
max_iters = 5000
device = 'cuda'
learning_rate = 3e-6
decay_lr = True
connection_layer = 12
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = max_iters - 100 # should be ~= max_iters per Chinchilla
min_lr = 3e-7 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
connection_read_layers_list = [0]
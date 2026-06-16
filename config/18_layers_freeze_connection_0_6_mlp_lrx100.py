from datetime import datetime
import torch
from pathlib import Path
name = Path(__file__).stem
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H")
out_name = f"{timestamp}_18_layers_freeze_connection_0_6_mlp_lrx100"
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
max_iters = 20000
device = 'cuda'
learning_rate = 3e-6 * 100
decay_lr = True
connection_layer = 12
connection_layer_mlp_enable = True
connection_read_layers_list = [0,6]
freeze = True
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = max_iters - 100 # should be ~= max_iters per Chinchilla
min_lr = 3e-7 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
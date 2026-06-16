import time
import torch
from datetime import date
 
eval_interval = 1000
eval_iters = 500
wandb_log = True # feel free to turn on
wandb_project = 'openwebtext-configs'
wandb_run_name = str(date.today()) + 'finetune-GPT2-10000iterations-complete-run'
out_dir = wandb_run_name
init_from = 'gpt2' # this is the largest GPT-2 model #original
always_save_checkpoint = True
batch_size = 12
gradient_accumulation_steps = 40
max_iters = 10000
device = 'cuda'
learning_rate = 3e-6
decay_lr = True
#freeze = True
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = max_iters - 1000 # should be ~= max_iters per Chinchilla
min_lr = 3e-7 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

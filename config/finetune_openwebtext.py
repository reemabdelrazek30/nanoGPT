import time
###### TODO : copy the file and edit it for grow version and overwrite the number of layers in the model
## EAGLE3 : Add connection to intermediate layers
import torch

#print(torch.version.hip)         # Should show something like '6.0.0'
#print(torch.cuda.is_available()) # True now!
#print(torch.cuda.get_device_name(0))  # Your AMD GPU name
#exit()
n_layer= 18
out_dir = 'out-openwebtext' + str(n_layer)
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'openwebtext-finetune'
#wandb_run_name = 'ft-mirror-' + str(time.time()) 
wandb_run_name = 'ft-10000-' + str(n_layer) 

init_from = 'gpt2' # this is the largest GPT-2 model #original
#init_from = 'gpt2' # changed to large due to memory constraints
# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 5000
device = 'cuda'
# finetune at constant LR
#learning_rate = 3e-6
decay_lr = True
#warmup_iters = 1000 # how many steps to warm up for
#lr_decay_iters = max_iters - 100 # should be ~= max_iters per Chinchilla
#min_lr = 3e-7 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
connection_layer = 13
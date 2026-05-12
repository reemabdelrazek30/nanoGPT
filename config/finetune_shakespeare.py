import time
###### TODO : copy the file and edit it for grow version and overwrite the number of layers in the model
## EAGLE3 : Add connection to intermediate layers
import torch

#print(torch.version.hip)         # Should show something like '6.0.0'
#print(torch.cuda.is_available()) # True now!
#print(torch.cuda.get_device_name(0))  # Your AMD GPU name
#exit()
n_layer=13
out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
#wandb_log = True # feel free to turn on
#wandb_project = 'shakespeare-mirror'
#wandb_run_name = 'ft-mirror-' + str(time.time()) 
#wandb_run_name = 'ft-mirror- ' + str(n_layer)

dataset = 'shakespeare'
init_from = 'gpt2' # this is the largest GPT-2 model #original
#init_from = 'gpt2' # changed to large due to memory constraints
# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20
device = 'cpu'
# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
connection_layer = 12

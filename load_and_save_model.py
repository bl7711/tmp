import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer from local path
model_path = "/raid/tmp/Llama-2-70b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set device map for 8 x H100 GPUs
device_map = "auto"  # Let the model automatically distribute across available GPUs
#device_map = {0: 'cuda:3', 1: 'cuda:5'}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    torch_dtype=torch.float16,  # Use half precision
    load_in_8bit=False,  # Disable 8-bit quantization
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Verify model distribution across GPUs
#print("Model device map:")
#for name, device in model.hf_device_map.items():
#    print(f"{name}: {device}")

save_path = "/raid/tmp/ckpt"
os.makedirs(save_path, exist_ok=True)
#torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

state_dict = model.state_dict()

# Split the state_dict based on the number of GPUs
# You will need to know which layers correspond to which GPU and manually split
# For simplicity, we assume 3 GPUs in the example, and the model layers are split across them.

# Define how many GPUs you have
num_gpus = torch.cuda.device_count()  # Get the number of available GPUs

# Get the layers for each GPU from the device map
gpu_layers = {i: [] for i in range(num_gpus)}
for layer_name, tensor in state_dict.items():
    # Determine which GPU this layer belongs to by checking the device map
    if 'model.layer' in layer_name:
        # Example: Split layer names by GPU
        gpu_index = int(layer_name.split('.')[2]) % num_gpus  # Assumes layers are split in order
        gpu_layers[gpu_index].append((layer_name, tensor))

# Save each GPU's layers to separate files
for gpu_index in range(num_gpus):
    gpu_state_dict = dict(gpu_layers[gpu_index])
    torch.save(gpu_state_dict, os.path.join(save_path, f"model_gpu{gpu_index}.pt"))
    print(f"Saved model state for GPU {gpu_index} to {os.path.join(save_path, f'model_gpu{gpu_index}.pt')}")

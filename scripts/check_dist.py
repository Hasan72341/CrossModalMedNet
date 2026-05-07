import torch
import os
from accelerate import Accelerator

def main():
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    accelerator = Accelerator()
    print(f"Rank {accelerator.local_process_index}: device {accelerator.device}, cuda={torch.cuda.current_device()}")

if __name__ == "__main__":
    main()

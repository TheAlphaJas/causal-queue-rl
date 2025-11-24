from torch.utils.tensorboard import SummaryWriter
import os
import time

def create_writer(log_dir, algo_name):
    ts = time.strftime("%Y%m%d-%H%M%S")
    full_dir = os.path.join(log_dir, f"{algo_name}_{ts}")
    os.makedirs(full_dir, exist_ok=True)
    writer = SummaryWriter(full_dir)
    return writer, full_dir

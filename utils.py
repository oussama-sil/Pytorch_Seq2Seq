import torch
import gc 
import psutil
import GPUtil


# To write with colors to the console 
class ConsoleColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def garbage_collect():
    """
        Clear the memory of the GPU and CPU 
    """
    torch.cuda.empty_cache()
    gc.collect()

def get_system_info():
    # CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)

    # RAM usage
    ram = psutil.virtual_memory()
    ram_percent = ram.percent

    # GPU information
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Assuming there is at least one GPU
            gpu_utilization = gpu.load * 100
            # print(gpu.memoryTotal-gpu.memoryFree)
            gpu_vram_percent = ((gpu.memoryTotal-gpu.memoryFree) / gpu.memoryTotal) * 100
        else:
            gpu_name = gpu_utilization = gpu_vram = gpu_vram_percent = None
    except Exception as e:
        print(f"Error while retrieving GPU information: {e}")
        gpu_name = gpu_utilization = gpu_vram = gpu_vram_percent = None

    return {
        "cpu": cpu_usage,
        "ram": ram_percent,
        "gpu": gpu_utilization,
        "vram": gpu_vram_percent
    }



def step_fct():
    """
        Function to execute each timestep to clear memory and display ressources usage
    """
    garbage_collect()
    sys_info=get_system_info()
    print(f'Sys info -> CPU:{sys_info["cpu"]:.2f}%  GPU:{sys_info["gpu"]:.2f}%  RAM:{sys_info["ram"]:.2f}%  VRAM:{sys_info["vram"]:.2f}%')



if __name__ == "__main__":
    step_fct()

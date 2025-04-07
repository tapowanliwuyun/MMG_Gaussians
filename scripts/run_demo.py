
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]
factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

excluded_gpus = set([])

source_dir = "/home/dkcs/mtgaussians_dataset"
dataset_dir = "mip360"
output_dir = "output_demo"
iteration_num = "30000"
dry_run = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {source_dir}/{dataset_dir}/{scene} -m {source_dir}/output/{dataset_dir}/{output_dir}/{scene} --eval -r {factor} --port {6009+int(gpu)}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {source_dir}/output/{dataset_dir}/{output_dir}/{scene} --iteration {iteration_num} --skip_train"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {source_dir}/output/{dataset_dir}/{output_dir}/{scene}"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set() 

    while jobs or future_to_job:
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
       
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job) 
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu) 

        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future) 
            gpu = job[0]  
            reserved_gpus.discard(gpu)  
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        time.sleep(5)
        
    print("All jobs have been processed.")

with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)


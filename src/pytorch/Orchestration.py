import subprocess
import os
from concurrent import futures

device = "1"
if device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = device

base_list = ["--model", "elmosl", "--epochs", "20", "--max_eval", "38", "--embed", "glove-retrofitting", "--earlystop", "5", "--save_model", "--spreadsheet", "--line"]
processes = [
    ["python", "SequenceLabelingElmo.py"] + base_list + ["--device", device, "--seed", "0"],
    ["python", "SequenceLabelingElmo.py"] + base_list + ["--device", device, "--seed", "1"],
    ["python", "SequenceLabelingElmo.py"] + base_list + ["--device", device, "--seed", "2"],
    ["python", "SequenceLabelingElmo.py"] + base_list + ["--device", device, "--seed", "3"],
    ["python", "SequenceLabelingElmo.py"] + base_list + ["--device", device, "--seed", "4"]
]

with futures.ProcessPoolExecutor(max_workers=2) as executor:
    results = executor.map(subprocess.call, processes)


print('completed.')
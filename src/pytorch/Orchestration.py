import subprocess
import os
from concurrent import futures

device = "3,5"
if device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = device

base_list = ["--model", "elmosl",
             "--epochs", "20",
             "--max_eval", "38",
             "--earlystop", "5",
             "--save_model",
             "--spreadsheet",
             "--line"]
processes = [
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + base_list + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + base_list + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + base_list + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + base_list + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + base_list + ["--device", device, "--seed", "4"]

    ["python", "SequencePointingBert.py", "--decode", "ordered"] + base_list + ["--device", device, "--seed", "0"],
    ["python", "SequencePointingBert.py", "--decode", "ordered"] + base_list + ["--device", device, "--seed", "1"],
    ["python", "SequencePointingBert.py", "--decode", "ordered"] + base_list + ["--device", device, "--seed", "2"],
    ["python", "SequencePointingBert.py", "--decode", "ordered"] + base_list + ["--device", device, "--seed", "3"],
    ["python", "SequencePointingBert.py", "--decode", "ordered"] + base_list + ["--device", device, "--seed", "4"],

    ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + base_list + ["--device", device, "--seed", "0"],
    ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + base_list + ["--device", device, "--seed", "1"],
    ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + base_list + ["--device", device, "--seed", "2"],
    ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + base_list + ["--device", device, "--seed", "3"],
    ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + base_list + ["--device", device, "--seed", "4"]

]

with futures.ProcessPoolExecutor(max_workers=8) as executor:
    results = executor.map(subprocess.call, processes)


print('completed.')
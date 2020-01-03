import subprocess
import os
from concurrent import futures
import argparse

parser = argparse.ArgumentParser(description='PASA Orchestration')
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--num_worker', default=1, type=int)
arguments = parser.parse_args()

device = arguments.device
if device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = device

train_base_list = ["--model", "bertptr",
                   "--epochs", "20",
                   "--max_eval", "38",
                   "--earlystop", "5",
                   "--save_model",
                   "--spreadsheet",
                   "--line"]

test_base_list = ["--spreadsheet",
                  "--save_output"]

processes = [
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + train_base_list + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + train_base_list + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + train_base_list + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + train_base_list + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingElmo.py", "--embed", "glove-retrofitting"] + train_base_list + ["--device", device, "--seed", "4"]

    # ["python", "SequencePointingBert.py", "--decode", "ordered"] + train_base_list + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--decode", "ordered"] + train_base_list + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--decode", "ordered"] + train_base_list + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--decode", "ordered"] + train_base_list + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--decode", "ordered"] + train_base_list + ["--device", device, "--seed", "4"]

    # ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + train_base_list + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + train_base_list + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + train_base_list + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + train_base_list + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--decode", "global_argmax"] + train_base_list + ["--device", device, "--seed", "4"]

    ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
        "--init_checkpoint", "../../results/pasa-bertsl-20191207-150951/model-0/epoch14-f0.8620.h5"],
    ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
        "--init_checkpoint", "../../results/pasa-bertsl-20191207-151017/model-0/epoch12-f0.8650_bert.h5"],
    ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
        "--init_checkpoint", "../../results/pasa-bertsl-20191207-151044/model-0/epoch10-f0.8611_bert.h5"],
    ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
        "--init_checkpoint", "../../results/pasa-bertsl-20191207-151112/model-0/epoch16-f0.8647_bert.h5"],
    ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
        "--init_checkpoint", "../../results/pasa-bertsl-20191207-151132/model-0/epoch12-f0.8631_bert.h5"],

    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191207-151236/model-0/epoch11-f0.8709_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191207-151258/model-0/epoch15-f0.8707_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191207-151316/model-0/epoch15-f0.8719_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191207-151338/model-0/epoch16-f0.8703_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191207-151354/model-0/epoch15-f0.8709_bert.h5"],

    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191214-092336/model-0/epoch11-f0.8702_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191214-092353/model-0/epoch15-f0.8703_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191220-073710/model-0/epoch19-f0.8718_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191214-092430/model-0/epoch13-f0.8698_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191220-073733/model-0/epoch15-f0.8706_bert.h5"],

    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191214-092106/model-0/epoch11-f0.8710_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191214-092116/model-0/epoch15-f0.8705_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191220-073635/model-0/epoch15-f0.8718_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191214-092205/model-0/epoch13-f0.8701_bert.h5"],
    ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    + test_base_list + ["--init_checkpoint",
                        "../../results/pasa-bertptr-20191220-073648/model-0/epoch15-f0.8707_bert.h5"]
]


with futures.ProcessPoolExecutor(max_workers=arguments.num_worker) as executor:
    results = executor.map(subprocess.call, processes)

print('completed.')
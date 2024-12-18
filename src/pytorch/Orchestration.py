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

train_base_list_bccwj = ["--epochs", "20",
                   "--max_eval", "38",
                   "--earlystop", "5",
                   "--save_model",
                   "--spreadsheet",
                   "--line",
                   "--with_bccwj"]

train_base_list_ntc = ["--epochs", "20",
                       "--max_eval", "38",
                       "--earlystop", "5",
                       "--save_model",
                       "--spreadsheet",
                       "--line"]

test_base_list = ["--spreadsheet",
                  "--save_output"]

processes = [
    # train sl bccwj
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_bccwj + ["--device", device, "--seed", "4"]

    # train spg bccwj
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "4"],

    # train spl bccwj
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "4"],

    # train spn bccwj
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed", "glove-retrofitting"] + train_base_list_bccwj + ["--device", device, "--seed", "4"],

    # train sl ntc
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list_ntc + ["--device", device, "--seed", "4"]

    # train spg ntc
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "4"]

    # train spl ntc
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "ordered", "--embed", "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train spn ntc
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
    #  "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
    #  "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
    #  "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
    #  "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
    #  "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "4"]

    # train bsl ntc
    # ["python", "SequenceLabelingBert.py", "--model", "bertsl"] + train_base_list_ntc + ["--device", device, "--seed", "0"]
    # ["python", "SequenceLabelingBert.py", "--model", "bertsl"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingBert.py", "--model", "bertsl"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingBert.py", "--model", "bertsl"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingBert.py", "--model", "bertsl"] + train_base_list_ntc + ["--device", device, "--seed", "4"]

    # train esl
    # ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list_bccwj + ["--device", device, "--seed", "4"]

    # train bspl
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "4"]

    # train bspg
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "4"]

    # train bsl bccwj
    # ["python", "SequenceLabelingBert.py", "--model", "bertsl"] + train_base_list_bccwj + ["--device", device, "--seed", "0"]

    # train luke ntc
    # ["python", "RunLuke.py", "--model", "luke", "--decode", "no_decoder"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingBert.py", "--model", "bertsl"] + train_base_list_ntc + ["--device", device, "--seed", "0"]

    # train GPT2 ntc, bccwj
    # ["python", "SequenceLabelingGPT2.py", "--model", "gpt2sl"] + train_base_list_ntc + ["--device", device, "--seed", "0"]
    # ["python", "SequenceLabelingGPT2.py", "--model", "gpt2sl"] + train_base_list_bccwj + ["--device", device, "--seed", "0"]

    # train GPT2 pointer ntc with WSC
    # ["python", "SequencePointingGPT2.py", "--model", "gpt2spn", "--decode", "no_decoder"] + train_base_list_ntc + ["--device", device, "--seed", "0"]
    ["python", "SequencePointingGPT2WSC.py", "--model", "gpt2spn", "--decode", "no_decoder"] + train_base_list_ntc + ["--device", device, "--seed", "0"]

    # train bspn with WSC
    # ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "no_decoder"] + train_base_list_ntc + ["--device", device, "--seed", "0"],

    # train GPT2 fine-tuning
    # ["python", "GPT2Finetuning.py", "--model", "gpt2ft"] + train_base_list_ntc + ["--device", device, "--seed", "0"]
    # ["python", "GPT2Finetuning.py", "--model", "gpt2ft"] + train_base_list_bccwj + ["--device", device, "--seed", "0"]

    # train GPT2 pre-training
    # ["python", "GPT2Pretraining.py", "--model", "gpt2pt"] + train_base_list_ntc + ["--device", device, "--seed", "0"]

    # train GPT2 ntc with ft GPT2
    # ["python", "SequenceLabelingGPT2.py", "--model", "gpt2sl"] + train_base_list_ntc + ["--device", device, "--seed", "0"]
    # ["python", "SequenceLabelingGPT2.py", "--model", "gpt2sl"] + train_base_list_bccwj + ["--device", device, "--seed", "0"]

    # train nictbsl ntc
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "4"]

    # train nictbspg ntc
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train nictbspl ntc
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train nictbspn ntc
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "no_decoder", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "no_decoder", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "no_decoder", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "no_decoder", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "no_decoder", "--with_db"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train nictbsl bccwj
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_bccwj + ["--device", device, "--seed", "4"]

    # train nictbspg bccwj
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspg", "--decode", "global_argmax"] + train_base_list_bccwj + ["--device", device, "--seed", "4"]

    # train nictbspl bccwj
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspl", "--decode", "ordered"] + train_base_list_bccwj + ["--device", device, "--seed", "4"]

    # train nictbspn bccwj
    # ["python", "SequencePointingBert.py", "--model", "nictbspn", "--decode", "no_decoder"] + train_base_list_bccwj + ["--device", device, "--seed", "0"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspn", "--decode", "no_decoder"] + train_base_list_bccwj + ["--device", device, "--seed", "1"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspn", "--decode", "no_decoder"] + train_base_list_bccwj + ["--device", device, "--seed", "2"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspn", "--decode", "no_decoder"] + train_base_list_bccwj + ["--device", device, "--seed", "3"],
    # ["python", "SequencePointingBert.py", "--model", "nictbspn", "--decode", "no_decoder"] + train_base_list_bccwj + ["--device", device, "--seed", "4"]

    # train sl ntc 5000 train data
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train sl ntc 10000 train data
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train sl ntc 20000 train data
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train sl ntc 40000 train data
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabeling.py", "--model", "lstm", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "4"]

    # train spg ntc 5000 train data
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "5000"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train spg ntc 10000 train data
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "10000"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train spg ntc 20000 train data
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "20000"] + train_base_list_ntc + ["--device", device, "--seed", "4"],

    # train spg ntc 40000 train data
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
    # ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed", "glove-retrofitting", "--num_data", "40000"] + train_base_list_ntc + ["--device", device, "--seed", "4"]

    # test sl ntc
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190414-135159/model-0/epoch17-f0.8438.h5"],
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190414-135227/model-0/epoch16-f0.8438.h5"],
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190414-134624/model-0/epoch16-f0.8465.h5"],
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190414-134659/model-0/epoch13-f0.8473.h5"],
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20190414-134727/model-0/epoch13-f0.8455.h5"],

    # # test spg ntc
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200314-175012-910005/model-0/epoch13-f0.8462.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200314-175025-120511/model-0/epoch10-f0.8469.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200314-173640-671684/model-0/epoch11-f0.8461.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200317-034712-773641/model-0/epoch9-f0.8466.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200317-054738-269770/model-0/epoch10-f0.8466.h5"],
    #
    # test spl ntc
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190605-161652/model-0/epoch12-f0.8440.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190605-161741/model-0/epoch10-f0.8455.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190605-161931/model-0/epoch9-f0.8476.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190605-162212/model-0/epoch11-f0.8455.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-pointer-20190605-162413/model-0/epoch12-f0.8456.h5"]

    # # test spn ntc
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200316-163641-708439/model-0/epoch14-f0.8459.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200316-163639-505130/model-0/epoch10-f0.8446.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200316-163231-161837/model-0/epoch10-f0.8450.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200316-163656-307044/model-0/epoch11-f0.8450.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20200320-004933-462343/model-0/epoch10-f0.8453.h5"],
    #
    # # test bsl ntc
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20191207-150951/model-0/epoch9-f0.8620.h5"],
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20191207-151017/model-0/epoch7-f0.8650.h5"],
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20191207-151044/model-0/epoch5-f0.8611.h5"],
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20191207-151112/model-0/epoch11-f0.8647.h5"],
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20191207-151132/model-0/epoch7-f0.8631.h5"],

    # test bsl ntc sep
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20210121-091206-686869/model-0/epoch7-f0.8644.h5"]
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20210221-184426-871180/model-0/epoch6-f0.8648.h5"]

    # test bsl ntc with sBert
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20210129-101751-974512/model-0/epoch7-f0.8640.h5"]
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20210203-141636-488318/model-0/epoch5-f0.8676.h5"]

    # test bsl ntc sep with noun
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertsl-20210305-084716-183613/model-0/epoch5-f0.8630.h5"]

    # # test bspg ntc
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191207-151236/model-0/epoch6-f0.8709.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191207-151258/model-0/epoch10-f0.8707.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191207-151316/model-0/epoch10-f0.8719.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191207-151338/model-0/epoch11-f0.8703.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191207-151354/model-0/epoch10-f0.8709.h5"],
    #
    # # test bspn ntc
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191214-092336/model-0/epoch6-f0.8702.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191214-092353/model-0/epoch10-f0.8703.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191220-073710/model-0/epoch17-f0.8718.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191214-092430/model-0/epoch8-f0.8698.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191220-073733/model-0/epoch10-f0.8706.h5"],
    #
    # # test bspl ntc
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191214-092106/model-0/epoch6-f0.8710.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191214-092116/model-0/epoch10-f0.8705.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191220-073635/model-0/epoch10-f0.8718.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191214-092205/model-0/epoch8-f0.8701.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
    # + test_base_list + ["--init_checkpoint",
    #                     "../../results/pasa-bertptr-20191220-073648/model-0/epoch10-f0.8707.h5"],
    #
    # # test sl bccwj
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191118-210943/model-0/epoch14-f0.7649.h5"],
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191120-014008/model-0/epoch12-f0.7646.h5"],
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191120-013713/model-0/epoch5-f0.7630.h5"],
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191120-013822/model-0/epoch14-f0.7686.h5"],
    # ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
    #  "--embed", "glove-retrofitting"] + test_base_list + [
    #     "--init_checkpoint", "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191120-014216/model-0/epoch12-f0.7641.h5"],
    #
    # # test spg bccwj
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191222-091011/model-0/epoch6-f0.7589.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191222-091034/model-0/epoch6-f0.7634.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191222-091047/model-0/epoch5-f0.7621.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191222-091049/model-0/epoch5-f0.7576.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191222-091147/model-0/epoch9-f0.7616.h5"],
    #
    # # test spn bccwj
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191229-015104/model-0/epoch5-f0.7614.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191228-205248/model-0/epoch5-f0.7584.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191228-205157/model-0/epoch13-f0.7628.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191229-014733/model-0/epoch14-f0.7619.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191228-205407/model-0/epoch7-f0.7615.h5"],
    #
    # # test spl bccwj
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191229-014829/model-0/epoch5-f0.7619.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191228-203829/model-0/epoch9-f0.7608.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191228-203835/model-0/epoch10-f0.7654.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191229-014905/model-0/epoch14-f0.7623.h5"],
    # ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
    #  "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-pointer-20191228-205223/model-0/epoch10-f0.7618.h5"],
    #
    # # test bsl bccwj
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
    # + test_base_list + ["--init_checkpoint", "../../results/pasa-bertsl-20191215-190021/model-0/epoch6-f0.7916.h5"],
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
    # + test_base_list + ["--init_checkpoint", "../../results/pasa-bertsl-20191217-070800/model-0/epoch8-f0.7918.h5"],
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
    # + test_base_list + ["--init_checkpoint", "../../results/pasa-bertsl-20191217-070804/model-0/epoch9-f0.7894.h5"],
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
    # + test_base_list + ["--init_checkpoint", "../../results/pasa-bertsl-20191217-070827/model-0/epoch8-f0.7916.h5"],
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
    # + test_base_list + ["--init_checkpoint", "../../results/pasa-bertsl-20191217-070856/model-0/epoch8-f0.7877.h5"],

    # test bsl bccwj sep
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
    # + test_base_list + ["--init_checkpoint", "../../results/pasa-bertsl-20210119-205041-733158/model-0/epoch8-f0.7906.h5"]

    # test bsl bccwj sep with noun
    # ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
    # + test_base_list + ["--init_checkpoint", "../../results/pasa-bertsl-20210305-084800-277814/model-0/epoch6-f0.7867.h5"]

    # # test bspn bccwj
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20191220-073831/model-0/epoch11-f0.7924.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20191220-073849/model-0/epoch12-f0.7922.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20191220-073908/model-0/epoch7-f0.7936.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20191220-073928/model-0/epoch11-f0.7945.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "no_decoder"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20191220-073944/model-0/epoch9-f0.7937.h5"],
    #
    # # test bspg bccwj
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200107-235157/model-0/epoch11-f0.7939.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200107-235217/model-0/epoch11-f0.7955.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200107-235238/model-0/epoch9-f0.7943.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200107-235316/model-0/epoch9-f0.7950.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "global_argmax"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200107-235337/model-0/epoch8-f0.7943.h5"],
    #
    # # test bspl bccwj
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200127-152715/model-0/epoch9-f0.7955.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200127-152723/model-0/epoch12-f0.7940.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200127-152722/model-0/epoch9-f0.7944.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200127-152714/model-0/epoch8-f0.7963.h5"],
    # ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
    #  "--decode", "ordered"] + test_base_list + [
    #     "--init_checkpoint", "../../results/pasa-bertptr-20200127-152713/model-0/epoch8-f0.7924.h5"]

    # combine results
    # ['python', 'CombinePrediction.py', '--event', 'acm', '--with_all']

    # Ensemble results
    # ['python', 'Ensemble.py', '--mode', 'all', '--corpus', 'all']

    # Segment analysis of Bccwj dataset
    # ['python', 'SegmentAnalysisBccwj.py', '--model', 'bsl_sep', '--reset_scores', '--with_initial_print', '--length_bin_size', '10', '--position_bin_size', '4'],

    # Sentence-length wise analysis of NTC dataset
    # ['python', 'SentenceLengthWiseAnalysisNtc.py', '--model', 'all', '--reset_scores', '--with_initial_print', '--length_bin_size', '10', '--position_bin_size', '4']
]


with futures.ProcessPoolExecutor(max_workers=arguments.num_worker) as executor:
    results = executor.map(subprocess.call, processes)

print('completed.')

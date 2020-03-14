import subprocess
import os
from concurrent import futures
import argparse

parser = argparse.ArgumentParser(description='PASA Orchestration')
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--num_worker', default=1, type=int)
parser.add_argument('--mode', default=None, type=str)
arguments = parser.parse_args()

device = arguments.device
if device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = device

train_base_list = ["--epochs", "20",
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

processes = {
    "train_sl_bccwj": [
        ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list + ["--device", device, "--seed", "0"],
        ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list + ["--device", device, "--seed", "1"],
        ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list + ["--device", device, "--seed", "2"],
        ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list + ["--device", device, "--seed", "3"],
        ["python", "SequenceLabeling.py", "--model", "lstm"] + train_base_list + ["--device", device, "--seed", "4"]],
    "train_spg_ntc": [
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "global_argmax", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "4"]],
    "train_spn_ntc": [
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
        ["python", "SequenceLabelingPointer.py", "--model", "pointer", "--decode", "no_decoder", "--embed",
         "glove-retrofitting"] + train_base_list_ntc + ["--device", device, "--seed", "4"]],
    "train_esl_ntc": [
        ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list + ["--device", device, "--seed",
                                                                                        "0"],
        ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list + ["--device", device, "--seed",
                                                                                        "1"],
        ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list + ["--device", device, "--seed",
                                                                                        "2"],
        ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list + ["--device", device, "--seed",
                                                                                        "3"],
        ["python", "SequenceLabelingElmo.py", "--model", "elmosl"] + train_base_list + ["--device", device, "--seed",
                                                                                        "4"]],

    "train_bspl_ntc": [
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list + [
            "--device", device, "--seed", "0"],
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list + [
            "--device", device, "--seed", "1"],
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list + [
            "--device", device, "--seed", "2"],
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list + [
            "--device", device, "--seed", "3"],
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "ordered"] + train_base_list + [
            "--device", device, "--seed", "4"]],

    "train_bspg_ntc": [
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list + [
            "--device", device, "--seed", "0"],
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list + [
            "--device", device, "--seed", "1"],
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list + [
            "--device", device, "--seed", "2"],
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list + [
            "--device", device, "--seed", "3"],
        ["python", "SequencePointingBert.py", "--model", "bertptr", "--decode", "global_argmax"] + train_base_list + [
            "--device", device, "--seed", "4"]],

    "train_nictbsl_ntc": [
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_ntc + ["--device", device,
                                                                                             "--seed", "0"],
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_ntc + ["--device", device,
                                                                                             "--seed", "1"],
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_ntc + ["--device", device,
                                                                                             "--seed", "2"],
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_ntc + ["--device", device,
                                                                                             "--seed", "3"],
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl"] + train_base_list_ntc + ["--device", device,
                                                                                             "--seed", "4"]],

    "train_nictbspg_ntc": [
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode",
         "global_argmax"] + train_base_list_ntc + ["--device", device, "--seed", "0"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode",
         "global_argmax"] + train_base_list_ntc + ["--device", device, "--seed", "1"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode",
         "global_argmax"] + train_base_list_ntc + ["--device", device, "--seed", "2"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode",
         "global_argmax"] + train_base_list_ntc + ["--device", device, "--seed", "3"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode",
         "global_argmax"] + train_base_list_ntc + ["--device", device, "--seed", "4"]],

    "train_nictbspl_ntc": [
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered"] + train_base_list_ntc + [
            "--device", device, "--seed", "0"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered"] + train_base_list_ntc + [
            "--device", device, "--seed", "1"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered"] + train_base_list_ntc + [
            "--device", device, "--seed", "2"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered"] + train_base_list_ntc + [
            "--device", device, "--seed", "3"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered"] + train_base_list_ntc + [
            "--device", device, "--seed", "4"]],

    "train_nictbspn_ntc": [
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder"] + train_base_list_ntc + [
            "--device", device, "--seed", "0"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder"] + train_base_list_ntc + [
            "--device", device, "--seed", "1"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder"] + train_base_list_ntc + [
            "--device", device, "--seed", "2"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder"] + train_base_list_ntc + [
            "--device", device, "--seed", "3"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder"] + train_base_list_ntc + [
            "--device", device, "--seed", "4"]],

    "train_nictbsl_bccwj": [
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_bccwj"] + train_base_list + ["--device",
                                                                                                         device,
                                                                                                         "--seed", "0"],
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_bccwj"] + train_base_list + ["--device",
                                                                                                         device,
                                                                                                         "--seed", "1"],
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_bccwj"] + train_base_list + ["--device",
                                                                                                         device,
                                                                                                         "--seed", "2"],
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_bccwj"] + train_base_list + ["--device",
                                                                                                         device,
                                                                                                         "--seed", "3"],
        ["python", "SequenceLabelingBert.py", "--model", "nictbsl", "--with_bccwj"] + train_base_list + ["--device",
                                                                                                         device,
                                                                                                         "--seed",
                                                                                                         "4"]],

    "train_nictbspg_bccwj": [
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "global_argmax",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "0"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "global_argmax",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "1"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "global_argmax",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "2"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "global_argmax",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "3"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "global_argmax",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "4"]],

    "train_nictbspl_bccwj": [
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "0"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "1"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "2"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "3"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "ordered",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "4"]],

    "train_nictbspn_bccwj": [
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "0"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "1"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "2"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "3"],
        ["python", "SequencePointingBert.py", "--model", "nictbsl", "--decode", "no_decoder",
         "--with_bccwj"] + train_base_list + ["--device", device, "--seed", "4"]],

    "test_bsl_ntc": [
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertsl-20191207-150951/model-0/epoch14-f0.8620.h5"],
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertsl-20191207-151017/model-0/epoch12-f0.8650.h5"],
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertsl-20191207-151044/model-0/epoch10-f0.8611.h5"],
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertsl-20191207-151112/model-0/epoch16-f0.8647.h5"],
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertsl-20191207-151132/model-0/epoch12-f0.8631.h5"]],

    "test_bspg_ntc": [
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191207-151236/model-0/epoch11-f0.8709.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191207-151258/model-0/epoch15-f0.8707.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191207-151316/model-0/epoch15-f0.8719.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191207-151338/model-0/epoch16-f0.8703.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "global_argmax", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191207-151354/model-0/epoch15-f0.8709.h5"]],

    "test_bspn_ntc": [
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191214-092336/model-0/epoch11-f0.8702.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191214-092353/model-0/epoch15-f0.8703.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191220-073710/model-0/epoch19-f0.8718.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191214-092430/model-0/epoch13-f0.8698.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "no_decoder", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191220-073733/model-0/epoch15-f0.8706.h5"]],

    "test_bspl_ntc": [
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191214-092106/model-0/epoch11-f0.8710.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191214-092116/model-0/epoch15-f0.8705.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191220-073635/model-0/epoch15-f0.8718.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191214-092205/model-0/epoch13-f0.8701.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--decode", "ordered", "--model", "bertptr"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertptr-20191220-073648/model-0/epoch15-f0.8707.h5"]],

    "test_sl_bccwj": [
        ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
         "--embed", "glove-retrofitting"] + test_base_list + [
            "--init_checkpoint",
            "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191118-210943/model-0/epoch19-f0.7649.h5"],
        ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
         "--embed", "glove-retrofitting"] + test_base_list + [
            "--init_checkpoint",
            "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191120-014008/model-0/epoch17-f0.7646.h5"],
        ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
         "--embed", "glove-retrofitting"] + test_base_list + [
            "--init_checkpoint",
            "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191120-013713/model-0/epoch10-f0.7630.h5"],
        ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
         "--embed", "glove-retrofitting"] + test_base_list + [
            "--init_checkpoint",
            "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191120-013822/model-0/epoch19-f0.7686.h5"],
        ["python", "EvalSequenceLabeling.py", "--device", device, "--model", "lstm", "--with_bccwj",
         "--embed", "glove-retrofitting"] + test_base_list + [
            "--init_checkpoint",
            "../../../PhD/projects/180630_oomorisan_PASA/results/pasa-lstm-20191120-014216/model-0/epoch17-f0.7641.h5"]],

    "test_spg_bccwj": [
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191222-091011/model-0/epoch11-f0.7589.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191222-091034/model-0/epoch11-f0.7634.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191222-091047/model-0/epoch10-f0.7621.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191222-091049/model-0/epoch10-f0.7576.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191222-091147/model-0/epoch14-f0.7616.h5"]],

    "test_spn_bccwj": [
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191229-015104/model-0/epoch10-f0.7614.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191228-205248/model-0/epoch10-f0.7584.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191228-205157/model-0/epoch18-f0.7628.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191229-014733/model-0/epoch19-f0.7619.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191228-205407/model-0/epoch12-f0.7615.h5"]],

    "test_spl_bccwj": [
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191229-014829/model-0/epoch10-f0.7619.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191228-203829/model-0/epoch9-f0.7608.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191228-203835/model-0/epoch15-f0.7654.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191229-014905/model-0/epoch19-f0.7623.h5"],
        ["python", "EvalSequenceLabelingPointer.py", "--device", device, "--model", "pointer", "--with_bccwj",
         "--embed", "glove-retrofitting", "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-pointer-20191228-205223/model-0/epoch15-f0.7618.h5"]],

    "test_bsl_bccwj": [
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertsl-20191215-190021/model-0/epoch11-f0.7916.h5"],
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertsl-20191217-070800/model-0/epoch13-f0.7918.h5"],
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertsl-20191217-070804/model-0/epoch14-f0.7894.h5"],
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertsl-20191217-070827/model-0/epoch13-f0.7916.h5"],
        ["python", "EvalSequenceLabelingBert.py", "--device", device, "--model", "bertsl", "--with_bccwj"]
        + test_base_list + ["--init_checkpoint",
                            "../../results/pasa-bertsl-20191217-070856/model-0/epoch13-f0.7877.h5"]],

    "test_bspn_bccwj": [
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20191220-073831/model-0/epoch16-f0.7924.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20191220-073849/model-0/epoch17-f0.7922.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20191220-073908/model-0/epoch12-f0.7936.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20191220-073928/model-0/epoch16-f0.7945.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "no_decoder"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20191220-073944/model-0/epoch14-f0.7937.h5"]],

    "test_bspg_bccwj": [
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200107-235157/model-0/epoch16-f0.7939.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200107-235217/model-0/epoch16-f0.7955.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200107-235238/model-0/epoch14-f0.7943.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200107-235316/model-0/epoch14-f0.7950.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "global_argmax"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200107-235337/model-0/epoch13-f0.7943.h5"]],

    "test_bspl_bccwj": [
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200127-152715/model-0/epoch14-f0.7955.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200127-152723/model-0/epoch17-f0.7940.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200127-152722/model-0/epoch14-f0.7944.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200127-152714/model-0/epoch13-f0.7963.h5"],
        ["python", "EvalSequencePointingBert.py", "--device", device, "--model", "bertptr", "--with_bccwj",
         "--decode", "ordered"] + test_base_list + [
            "--init_checkpoint", "../../results/pasa-bertptr-20200127-152713/model-0/epoch13-f0.7924.h5"]],

    # combine results
    "combine_results": [
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "sl", "--corpus", "ntc", "--emb", "glove"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spg", "--corpus", "ntc", "--emb", "glove"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spl", "--corpus", "ntc", "--emb", "glove"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spn", "--corpus", "ntc", "--emb", "glove"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "sl", "--corpus", "ntc", "--emb", "bert"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spg", "--corpus", "ntc", "--emb", "bert"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spl", "--corpus", "ntc", "--emb", "bert"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spn", "--corpus", "ntc", "--emb", "bert"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "sl", "--corpus", "bccwj", "--emb", "glove"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spg", "--corpus", "bccwj", "--emb", "glove"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spl", "--corpus", "bccwj", "--emb", "glove"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spn", "--corpus", "bccwj", "--emb", "glove"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "sl", "--corpus", "bccwj", "--emb", "bert"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spg", "--corpus", "bccwj", "--emb", "bert"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spl", "--corpus", "bccwj", "--emb", "bert"],
        ["python", "CombinePrediction.py", "--event", "acm", "--model", "spn", "--corpus", "bccwj", "--emb", "bert"]]
}

proc = []
for tag in arguments.mode.split(","):
    proc += processes[tag]
with futures.ProcessPoolExecutor(max_workers=arguments.num_worker) as executor:
    results = executor.map(subprocess.call, processes)

print('completed.')

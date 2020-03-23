#!/usr/bin/env python3
import json
import os
import subprocess

input_checkpoint = 'imagenet_checkpoint/'
dataset_index_file = "/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt"
class_names_file = '/data/sample-models/resources/data/imagenet/imagenet1000.names'
preprocessor = 'imagenet_caffe'
input_names = 'input_1'
output_names = 'probs/Softmax'
input_shapes = '1,224,224,3'

dry_run=True

FRAMEWORKS = ['tflite', 'TF', 'TVM'] # columns
PRECISIONS = ['FP32', 'INT8'] # rows
COMPRESSION_MODES = ['Baseline', 'LEIP'] # rows

commands_run = []
current_section = None
current_framework = None
current_precision = None
current_compression = None


def logCmd(args):
    commands_run.append(args)
    if not dry_run:
        subprocess.check_call(args)
sections = []
section_to_results = {}
section_to_name = {}
def getResults(dirname):
    global section_to_results
    global current_framework
    global current_precision
    global current_compression

    resultPath = os.path.join(dirname, 'results.json')
    data = json.loads(open(resultPath, 'r').read())
    #print(data)

    if current_framework and current_precision and current_compression:
        section_to_results[(current_framework, current_precision, current_compression,)] = data

def setSectionName(name, framework=None, precision=None, compression=None):
    global current_section
    global sections
    global current_framework
    global current_precision
    global current_compression

    if framework is not None:
        if framework not in FRAMEWORKS:
            raise Exception('bad framework '+framework)
    if precision is not None:
        if precision not in PRECISIONS:
            raise Exception('bad precision ' + precision)
    if compression is not None:
        if compression not in COMPRESSION_MODES:
            raise Exception('bad compression ' + compression)

    if framework and precision and compression:
        current_framework = framework
        current_precision = precision
        current_compression = compression
    else:
        current_framework = None
        current_precision = None
        current_compression = None

    if name is None:
        name = "{} {} {}".format(framework, precision, compression)

    section_to_name[(current_framework, current_precision, current_compression,)] = name
    current_section = name
    sections.append(name)
    commands_run.append(['# '+name])

setSectionName("Preparation")

logCmd(["rm", "-rf", "variants", "baselineFp32Results"])
logCmd(["mkdir", "variants"])
logCmd(["mkdir", "baselineFp32Results"])

setSectionName(None, "TF", "FP32", "Baseline")

logCmd(["leip", "evaluate", "--output_path","baselineFp32Results", "--framework", "tf2", "--input_path", input_checkpoint, "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor, "--input_shapes", input_shapes, "--input_names", input_names, "--output_names", output_names])
getResults("baselineFp32Results/")
setSectionName("LEIP Compress")

logCmd(["leip", "compress", "--input_path", input_checkpoint, "--quantizer", "ASYMMETRIC", "--bits", "8", "--output_path", "variants/checkpointCompressed/"])
logCmd(["leip", "compress", "--input_path", input_checkpoint, "--quantizer", "POWER_OF_TWO", "--bits", "8", "--output_path", "variants/checkpointCompressedPow2/"])

setSectionName(None, "TF", "FP32", "LEIP")
logCmd(["leip", "evaluate", "--output_path","variants/checkpointCompressed/", "--framework", "tf2", "--input_path", "variants/checkpointCompressed/model_save/", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor, "--input_shapes", input_shapes, "--input_names", input_names, "--output_names", output_names])
getResults("variants/checkpointCompressed/")

setSectionName(None, "TVM", "INT8", "Baseline")
logCmd(["rm", "-rf", "variants/compiled_tvm_int8"])
logCmd(["mkdir", "variants/compiled_tvm_int8"])
logCmd(["leip", "compile", "--input_path", input_checkpoint, "--input_shapes", input_shapes, "--output_path", "variants/compiled_tvm_int8/bin", "--input_types=uint8", "--data_type=int8"])
logCmd(["leip", "evaluate", "--output_path","variants/compiled_tvm_int8/","--framework", "tvm", "--input_names", input_names, "--input_types=uint8", "--input_shapes", input_shapes, "--input_path", "variants/compiled_tvm_int8/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
getResults("variants/compiled_tvm_int8/")

setSectionName(None, "TVM", "FP32", "Baseline")
logCmd(["rm", "-rf", "variants/compiled_tvm_fp32"])
logCmd(["mkdir", "variants/compiled_tvm_fp32"])
logCmd(["leip", "compile", "--input_path", input_checkpoint, "--input_shapes", input_shapes, "--output_path", "variants/compiled_tvm_fp32/bin", "--input_types=float32", "--data_type=float32"])
logCmd(["leip", "evaluate", "--output_path","variants/compiled_tvm_fp32/","--framework", "tvm", "--input_names", input_names, "--input_types=float32", "--input_shapes", input_shapes, "--input_path", "variants/compiled_tvm_fp32/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
getResults("variants/compiled_tvm_fp32/")

setSectionName(None, "TVM", "INT8", "LEIP")
logCmd(["rm", "-rf", "variants/leip_compiled_tvm_int8"])
logCmd(["mkdir", "variants/leip_compiled_tvm_int8"])
logCmd(["leip", "compile", "--input_path", "variants/checkpointCompressed/model_save/", "--input_shapes", input_shapes, "--output_path", "variants/leip_compiled_tvm_int8/bin", "--input_types=uint8", "--data_type=int8"])
logCmd(["leip", "evaluate", "--output_path","variants/leip_compiled_tvm_int8","--framework", "tvm", "--input_names", input_names, "--input_types=uint8", "--input_shapes", input_shapes, "--input_path", "variants/leip_compiled_tvm_int8/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
getResults("variants/leip_compiled_tvm_int8/")

setSectionName(None, "TVM", "FP32", "LEIP")
logCmd(["rm", "-rf", "variants/leip_compiled_tvm_fp32"])
logCmd(["mkdir", "variants/leip_compiled_tvm_fp32"])
logCmd(["leip", "compile", "--input_path", "variants/checkpointCompressed/model_save/", "--input_shapes", input_shapes, "--output_path", "variants/leip_compiled_tvm_fp32/bin", "--input_types=float32", "--data_type=float32"])
logCmd(["leip", "evaluate", "--output_path","variants/leip_compiled_tvm_fp32","--framework", "tvm", "--input_names", input_names, "--input_types=float32", "--input_shapes", input_shapes, "--input_path", "variants/leip_compiled_tvm_fp32/bin", "--test_path="+dataset_index_file, "--class_names="+class_names_file, "--task=classifier", "--dataset=custom", "--preprocessor", preprocessor])
getResults("variants/leip_compiled_tvm_fp32/")



output_rows = [''] + FRAMEWORKS # columns

ROWTYPES = ['Inference Speed', 'Accuracy']

for rowtype in ROWTYPES:
    for compression_mode in COMPRESSION_MODES:
        for precision in PRECISIONS:
            row = []
            output_rows.append(row)
            rowname = "{} {} {}".format(compression_mode, precision, rowtype)
            row.append(rowname)
            for framework in FRAMEWORKS: # column

                section = (framework, precision, compression_mode,)
                section_name = "{} {} {}".format(framework, precision, compression_mode)

                if section in section_to_results:

                    results = section_to_results[section]
                    #print(results)
                    top1 = results['results']['evaluate']['results']['top1']
                    top5 = results['results']['evaluate']['results']['top5']
                    items = results['results']['evaluate']['results']['items']
                    duration = results['results']['evaluate']['results']['duration']

                    per_sec = items / duration
                    #print('{}\t{}\t{}\t{}'.format(section_name, top1, top5, per_sec))
                    if rowtype == 'Inference Speed':
                        cellvalue = str(per_sec) + 'inferences/sec'
                    elif rowtype == 'Accuracy':
                        cellvalue = "Top1: {}\nTop5: {}".format(top1, top5)
                    else:
                        cellvalue = '???'
                else:
                    print('No results for {}'.format(section_name))
                    cellvalue = 'N/A'
                row.append(cellvalue)
    #print(section_to_results)

print('Output report:')
for command in commands_run:
    line = ' '.join(command)
    print(line)

for row in output_rows:
    rowstr = "\t".join(row)
    print(rowstr)

from timeit import default_timer as timer
import os, sys
import argparse
import lzma
import subprocess
import csv
import matplotlib
import matplotlib.pyplot as plt
import math

print("\t\t Parallel Quantum Simulation on GPUs\n")

parser = argparse.ArgumentParser("gpu_benchmark")
parser.add_argument('-d', '--benchdir', help="Directory to read benchmarks.", default='')
parser.add_argument('-c', '--codedir', help="Directory to code.", default='../')
parser.add_argument('-g', '--kernelconfig', help="Path of kernel configuration", default='../kernel.config')
parser.add_argument('-r', '--resultsdir', help="Directory to store results.", default='results/gpu')
parser.add_argument('-n', '--nsamples', help="Number of samples", default=2)
parser.add_argument('-m', '--max', help="Maximum qubits", default=0)
args = parser.parse_args()

bench_dir = args.benchdir
code_dir = args.codedir
main_logs_dir = args.resultsdir
kernelconfig = args.kernelconfig
nsamples = int(args.nsamples)
max_qubits = int(args.max)

header = ['Circuit', 'Initial time', 'Schedule time', 'Transfer time', 'Simulation time', 
          'Power consumption', 'Energy consumption',
          'Tableau partitions', 'Tableau memory', 'Tableau step speed', 
          'Circuit memory', 'Average of parallel gates', 'Clifford gates']

# configs = ['-overlap', '-no-overlap', '-sync']
# log_dirs = ['overlap', 'no_overlap', 'sync']

configs = ['-no-overlap']
log_dirs = ['default']

config2dir = dict()
config2csv = dict()
config2res = dict()

def compile_clean():
    args = ('make', '-C', code_dir, 'clean')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    args = ('make', '-C', code_dir, 'nocolor=1')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

def write_log_file(output, log_file):
    with open(log_file, 'w') as log_f:
        log_f.write(str(output))
        log_f.close()

def average_results(sample, in_cell, in_entry, run):
    for i, cell in enumerate(header):
        if (cell == in_cell):
            if ('time' in in_cell or 'speed' in in_cell or 'consumption'):
                sample[i] += float(in_entry)
                if (run == nsamples - 1):
                    sample[i] /= nsamples
            else:
                sample[i] = float(in_entry)
            break

def run_config(circuit, config, log_dir, benchpath):
    verbose_opt = '--verbose=0'
    kernelconfig_opt = '--config-path=' + kernelconfig
    binaryfile = code_dir + '/quasarq'
    args = (binaryfile, benchpath, config, verbose_opt, kernelconfig_opt)
    row = [circuit]
    avg = [0] * len(header)
    for i in range(0, nsamples):
        print("Running [%-12s] with configuration [%-11s] the %d-time... " %(circuit, config, i+1), end='\r'), sys.stdout.flush()
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        log_file = log_dir + '/' + 'log_run_' + str(i) + '_' + circuit + '.txt'
        output = popen.stdout.read().decode('utf-8')
        write_log_file(output, log_file)
        outputlines = output.splitlines()
        for line in outputlines:
            if line.strip()[0] == '-':
                break
            if 'Read' in line:
                continue
            splitline = line.split(':')
            if len(splitline) < 2:
                continue
            value = splitline[1]
            if ('msec' in value):
                value = value.removesuffix('msec').strip()
            elif ('MB' in value):
                value = value.removesuffix('MB').strip()
            elif ('GB/sec' in value):
                value = value.removesuffix('GB/sec').strip()
            elif ('GB' in value):
                value = value.removesuffix('GB').strip()
            elif ('watt' in value):
                value = value.removesuffix('watt').strip()
            elif ('joules' in value):
                value = value.removesuffix('joules').strip()
            cell = splitline[0].strip()
            average_results(avg, cell, value, i)
    csv_writer = csv.writer(config2csv[config])
    for i in range(1, len(header)):
        rounded = "%.2f" % avg[i]
        row.append(rounded)
    csv_writer.writerow(row)

def benchmark():
    print("Compiling directory %s from scratch... " %(code_dir), end=''), sys.stdout.flush()
    compile_clean()
    print("done.\n")

    for dir in log_dirs:
        new_dir = main_logs_dir + '/' +  dir
        os.makedirs(new_dir, exist_ok=True)

    for i, config in enumerate(configs):
        log_dir = main_logs_dir + '/'  + log_dirs[i]
        config2dir.update({ config: log_dir })
        csv_path = log_dir + '/' + log_dirs[i] + '.csv'
        f = open(csv_path, 'a', encoding='UTF8', newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        config2csv.update({ config: f })

    file_list = [f for f in os.listdir(bench_dir) if f.endswith(".xz")]
    file_list.sort()
    for i,file_name in enumerate(file_list):
        path = os.path.join(bench_dir, file_name)
        with lzma.open(path, mode='rt', encoding='utf-8') as f:
            stim_string_file_name = os.path.splitext(file_name)
            file_name_stim = os.path.splitext(stim_string_file_name[0])
            written_file_path = file_name_stim[0] + '_tmp' + file_name_stim[1]
            if not os.path.exists(written_file_path):
                lines = list()
                for line in f:
                    if '#' in line:
                        qubits = int(line.replace('#', ''))
                        if (max_qubits and qubits > max_qubits):
                            written_file_path = ''
                            break
                    lines.append(line)
                if (written_file_path == ''):
                    continue   
                stim = ''.join(lines)       
                with open(written_file_path, 'w') as str_f:
                    str_f.write(stim)
                str_f.close()
            for config in config2dir.keys():
                run_config(file_name_stim[0], config, config2dir[config], written_file_path)
            os.remove(written_file_path)

    for config in config2csv.keys():
        config2csv[config].close()

    print("Finished %d circuits with %d configurations for %d times.%40s\n" %(len(file_list), len(configs), nsamples, " ")), sys.stdout.flush()

benchmark()
from timeit import default_timer as timer
import os, sys
import argparse
import lzma
import subprocess
import csv
import matplotlib
import matplotlib.pyplot as plt
import math

print("\t\t Design choices in QuaSARQ\n")

parser = argparse.ArgumentParser("gpu_benchmark")
parser.add_argument('-max', '--max', help="Maximum qubits to benchmarks.", default=2000)
parser.add_argument('-min', '--min', help="Minimum qubits to benchmarks.", default=1000)
parser.add_argument('-step', '--step', help="Step size.", default=1000)
parser.add_argument('-c', '--codedir', help="Directory to code.", default='src')
parser.add_argument('-g', '--kernelconfig', help="Path of kernel configuration", default='src/kernel.config')
parser.add_argument('-r', '--resultsdir', help="Directory to store results.", default='results/quasarq/configs')
parser.add_argument('-n', '--nsamples', help="Number of samples", default=2)
args = parser.parse_args()

max_qubits = args.max
min_qubits = args.min
step_qubits = args.step
code_dir = args.codedir
main_logs_dir = args.resultsdir
kernelconfig = args.kernelconfig
nsamples = int(args.nsamples)

header = ['Circuit', 'Initial time', 'Schedule time', 'Transfer time', 'Simulation time', 
          'Power consumption', 'Energy consumption',
          'Tableau partitions', 'Tableau memory', 'Tableau step speed', 
          'Circuit memory', 'Average of parallel gates', 'Clifford gates']

make_opt='cinterleave=0'

configs = ['--H=.125 --S=.125', '--H=.25 --S=.25', '--H=.375 --S=.375']
log_dirs = ['no-interleave_p25', 'no-interleave_p50', 'no-interleave_p75']

config2dir = dict()
config2csv = dict()
config2res = dict()

def compile_clean():
    args = ('make', '-C', code_dir, 'clean')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    args = ('make', '-C', code_dir, 'nocolor=1', make_opt)
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

def run_config(qubits, config, log_dir):
    verbose_opt = '--verbose=0'
    kernelconfig_opt = '--config-path=' + kernelconfig
    qubits_opt = '--qubits=' + str(qubits)
    binaryfile = code_dir + '/quasarq'
    args = (binaryfile, qubits_opt, verbose_opt, kernelconfig_opt, config)
    circuit = 'q' + str(qubits)
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

    qubits_range = [q for q in range(int(min_qubits), int(max_qubits) + 1, int(step_qubits))]
    qubits_range.sort()
    for q in qubits_range:
        for config in config2dir.keys():
            run_config(q, config, config2dir[config])

    for config in config2csv.keys():
        config2csv[config].close()

    print("Finished %d circuits with %d configurations for %d times.%40s\n" %(len(qubits_range), len(configs), nsamples, " ")), sys.stdout.flush()

benchmark()
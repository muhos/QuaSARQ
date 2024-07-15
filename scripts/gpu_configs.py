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
parser.add_argument('-max', '--max', help="Maximum qubits to benchmarks.", default=10000)
parser.add_argument('-min', '--min', help="Minimum qubits to benchmarks.", default=1000)
parser.add_argument('-step', '--step', help="Step size.", default=1000)
parser.add_argument('-c', '--codedir', help="Directory to code.", default='../')
parser.add_argument('-g', '--kernelconfig', help="Path of kernel configuration", default='../kernel.config')
parser.add_argument('-r', '--resultsdir', help="Directory to store results.", default='results/gpu')
parser.add_argument('-n', '--nsamples', help="Number of samples", default=2)
parser.add_argument('-p', '--plot', help="Plot only data saved in csv files", default=0)
args = parser.parse_args()

max_qubits = args.max
min_qubits = args.min
step_qubits = args.step
code_dir = args.codedir
main_logs_dir = args.resultsdir
kernelconfig = args.kernelconfig
nsamples = int(args.nsamples)
plot_data = int(args.plot)

header = ['Circuit', 'Initial time', 'Schedule time', 'Transfer time', 'Simulation time', 
          'Power consumption', 'Energy consumption',
          'Tableau partitions', 'Tableau memory', 'Tableau step speed', 
          'Circuit memory', 'Average of parallel gates', 'Clifford gates']

make_opt='winterleave=1'

configs = ['interleave_words']
log_dirs = ['interleave_words']

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
    args = (binaryfile, qubits_opt, verbose_opt, kernelconfig_opt)
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

def plot():
    for i, config in enumerate(configs):
        log_dir = main_logs_dir + '/'  + log_dirs[i]
        csv_path = log_dir + '/' + log_dirs[i] + '.csv'
        if (not os.path.exists(csv_path)):
            print("%s does not exist." % csv_path)
            return
        csv_reader = csv.reader(open(csv_path, 'r'), delimiter=",")
        next(csv_reader)
        data_per_config = []
        for line in csv_reader:
            data_per_config.append(math.log10(float(line[3]) + float(line[4])))
        config2res.update({ config: data_per_config })
    xLab = 'Clifford circuits (1,000 - 500,000 qubits)'
    yLab = 'Run Time (milliseconds)'
    fontSize=50
    mS=10
    mW=2
    lW=1.5
    step = 50
    start = 0
    models = dict()
    for config in configs:
        models.update({ config: list(range(start, len(config2res[config]))) })
    fig = plt.figure(figsize=(25,13))
    #plt.rcParams["font.family"] = "Times New Roman"
    # plt.plot(models['-sync'],  config2res['-sync'], linestyle='-', color='tab:red', marker='.', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    # plt.plot(models['-no-overlap'],  config2res['-no-overlap'], linestyle='-', color='tab:blue', marker='+', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    # plt.plot(models['-overlap'],  config2res['-overlap'], linestyle='-', color='tab:green', marker='*', markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
    plt.scatter(models['-sync'],   config2res['-sync'], color='tab:red',  	marker=',', s=mS, 	linewidths=lW, zorder=1)
    plt.scatter(models['-overlap'],   config2res['-overlap'], color='tab:green',  	marker=',', s=mS, 	linewidths=lW, zorder=1)
    leg = plt.legend(['sync', 'no-overlap', 'overlap'], fontsize=40)
    leg.get_frame().set_linewidth(4.0)
    plt.yticks(fontsize=fontSize)
    plt.xticks(fontsize=fontSize)
    ax = plt.gca()
    ax.xaxis.get_major_ticks()[0].set_visible(False)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    ax.tick_params(axis="y",direction="in", length=8, right=True)
    ax.tick_params(axis="x",direction="in", length=8, top=True)
    plt.xlabel(xLab, fontsize=fontSize, fontweight='normal', labelpad=25)
    plt.ylabel(yLab, fontsize=fontSize, fontweight='normal', labelpad=25)
    plt.tight_layout(pad=2)
    plt.savefig(main_logs_dir + "/gpu_configs.pdf", dpi=1000)

benchmark()
#################################
### Design choices in QuaSARQ ###
#################################

import os, sys
import argparse
import subprocess
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as font_manager
import argparse

Path = os.path

main_dir = Path.dirname(Path.dirname(Path.abspath(__file__))) + '/'

parser = argparse.ArgumentParser("gpu_benchmark")
parser.add_argument('-max', '--max', help="Maximum qubits to benchmarks.", default=2000)
parser.add_argument('-min', '--min', help="Minimum qubits to benchmarks.", default=1000)
parser.add_argument('-step', '--step', help="Step size.", default=1000)
parser.add_argument('-src', '--src', help="Directory to source code.", default='src')
parser.add_argument('-o', '--output', help="Directory to store results.", default='results/quasarq/configs')
parser.add_argument('-g', '--kernelconfig', help="Path of kernel configuration", default='src/kernel.config')
parser.add_argument('-n', '--nsamples', help="Number of samples", default=2)
parser.add_argument('--ccmix', help="Perform column-interleaving experiments using different number of columns", action=argparse.BooleanOptionalAction)
parser.add_argument('--cpmix', help="Perform column-interleaving experiments using different probabilities of H and S gates", action=argparse.BooleanOptionalAction)
parser.add_argument('--wmix', help="Perform word-interleaving experiments", action=argparse.BooleanOptionalAction)
parser.add_argument('--wsize', help="Perform word-size experiments", action=argparse.BooleanOptionalAction)
parser.add_argument('--plotonly', help="Plot only assuming csv files exist", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

code_dir = main_dir + args.src
main_logs_dir = main_dir + args.output
kernelconfig = main_dir + args.kernelconfig
max_qubits = int(args.max)
min_qubits = int(args.min)
step_qubits = int(args.step)
nsamples = int(args.nsamples)

qubits_range = [q for q in range(int(min_qubits), int(max_qubits) + 1, int(step_qubits))]
qubits_range.sort()

header = ['Circuit', 'Initial time', 'Schedule time', 'Transfer time', 'Simulation time', 
          'Power consumption', 'Energy consumption',
          'Tableau partitions', 'Tableau memory', 'Tableau step speed', 
          'Circuit memory', 'Average parallel gates', 'Clifford gates']

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

def run_config(qubits, make_config, run_config, log_dir):
    verbose_opt = '--verbose=0'
    kernelconfig_opt = '--config-path=' + kernelconfig
    qubits_opt = '--qubits=' + str(qubits)
    binaryfile = code_dir + '/quasarq'
    args = (binaryfile, qubits_opt, verbose_opt, kernelconfig_opt, ('' if run_config == 'default' else run_config))
    circuit = 'q' + str(qubits)
    avg = [0] * len(header)
    for i in range(0, nsamples):
        print(" Running [%-12s] with configuration [%-11s / %11s] the %d-time... " %(circuit, make_config, run_config, i+1), end='\r'), sys.stdout.flush()
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        log_file = log_dir + '/' + 'log_run_' + str(i) + '_' + circuit + '.txt'
        output = popen.stdout.read().decode('utf-8')
        write_log_file(output, log_file)
        outputlines = output.splitlines()
        for line in outputlines:
            if 'Statistics' in line:
                continue
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
    return avg


def make_clean(make_config):
    args = ('make', '-C', code_dir, 'clean')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    args = ('make', '-C', code_dir, 'nocolor=1', make_config)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

def add_config(config2dir, config2csv, config):
    log_dir = main_logs_dir + '/'  + config
    os.makedirs(log_dir, exist_ok=True)
    config2dir.update({ config: log_dir })
    csv_path = log_dir + '/' + config + '.csv'
    f = open(csv_path, 'a' if args.plotonly else 'w', encoding='UTF8', newline='')
    csv_writer = csv.writer(f)
    if not args.plotonly: csv_writer.writerow(header)
    config2csv.update({ config: f })

def write_csv(config2csv, config, circuit, results):
    csv_writer = csv.writer(config2csv[config])
    row = [circuit]
    for i in range(1, len(header)):
        rounded = "%.2f" % results[i]
        row.append(rounded)
    csv_writer.writerow(row)

def run(config2dir, config2csv, make_config, config, config_name):
    add_config(config2dir, config2csv, config_name)
    if not args.plotonly:
        make_clean(make_config)
        for q in qubits_range:
            results = run_config(q, make_config, config, config2dir[config_name])
            write_csv(config2csv, config_name, 'q' + str(q), results)
    config2csv[config_name].close()

def collect(data, config2csv):
    data.clear()
    table = dict()
    for config in config2csv.keys():
        table.update({ config: [[]] })
    for config, csv_path in config2csv.items():
        path = csv_path.name
        if (not os.path.exists(path)):
            raise Exception("%s does not exist." % path)
        csv_data = list(csv.reader(open(path, 'r'), delimiter=","))
        table.update({ config: csv_data })
    start = int(min_qubits / 1000)
    end = int(max_qubits / 1000)
    for config in config2csv.keys():
        time = list()
        for row in table[config]:
            if row[0] == 'Circuit': continue
            assert(len(row) > 3)
            rowtime = float(row[4])
            time.append(rowtime)
        time.sort()
        time = time[start-start:(end-start)+1]
        data.update({ config: (list(range(start, end + 1)), time) })

def show_plot(output, legend):
    fontSize=70; legFontSize=65; fontWeight='bold'
    xLab = 'Qubits in thousands'
    yLab = 'Run time (milliseconds)'
    plt.yticks(fontsize=fontSize, fontweight=fontWeight)
    plt.xticks(fontsize=fontSize, fontweight=fontWeight)
    leg = plt.legend(legend, prop=font_manager.FontProperties(weight='bold', size=legFontSize))
    leg.get_frame().set_linewidth(4.0)
    ax = plt.gca()
    ax.xaxis.get_major_ticks()[0].set_visible(False)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    ax.tick_params(axis="y",direction="in", length=8, right=True)
    ax.tick_params(axis="x",direction="in", length=8, top=True)
    plt.xlabel(xLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
    plt.ylabel(yLab, fontsize=fontSize, fontweight=fontWeight, labelpad=25)
    plt.tight_layout(pad=2)
    plt.savefig(output + '.pdf', dpi=1000)

    
def benchmark():

    config2dir = dict()
    config2csv = dict()
    data = dict()

    mS=30; mW=4; lW=6

    print("")

    if args.wsize:
        print("Experimenting with different word sizes..")
        make_configs=['word=8',     'word=32',     'word=64']
        legend =     ['8-bit word', '32-bit word', '64-bit word']
        colors =     ['tab:red',    'tab:blue',    'tab:green']
        markers =    ['+',           '.',          '*']
        config2dir.clear()
        config2csv.clear()
        for make_config in make_configs:
            config_name = make_config.replace('=', '_c')
            run(config2dir, config2csv, make_config, 'default', config_name)
        print(" Finished %d circuits with %d configurations for %d times.%50s\n" %(len(qubits_range), len(make_configs), nsamples, " ")), sys.stdout.flush()
        collect(data, config2csv)
        plt.figure(figsize=(25,13))
        plt.rcParams["font.family"] = "Times New Roman"
        for i, config in enumerate(data.keys()):
            plt.plot(data[config][0],  data[config][1], linestyle='-', color=colors[i], marker=markers[i], markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
        show_plot(main_logs_dir + '/word_sizes_plot', legend)
        
        
    if args.wmix:
        print("Experimenting with word-interleaving..")
        make_configs=['winterleave=0',    'winterleave=1']
        legend =     ['separate X and Z', 'XZ 1-word interleave']
        colors =     ['tab:green',        'tab:blue']
        markers =    ['*',                '+']
        config2dir.clear()
        config2csv.clear()
        for make_config in make_configs:
            config_name = make_config.replace('=', '_')
            run(config2dir, config2csv, make_config, 'default', config_name)
        print(" Finished %d circuits with %d configurations for %d times.%50s\n" %(len(qubits_range), len(make_configs), nsamples, " ")), sys.stdout.flush()
        collect(data, config2csv)
        plt.figure(figsize=(25,13))
        plt.rcParams["font.family"] = "Times New Roman"
        for i, config in enumerate(data.keys()):
            plt.plot(data[config][0],  data[config][1], linestyle='-', color=colors[i], marker=markers[i], markeredgewidth=mW, markersize=mS, fillstyle='full', linewidth=lW)
        show_plot(main_logs_dir + '/word_interleaving_plot', legend)


    if args.ccmix:
        print("Experimenting with column-interleaving..")
        make_configs=['cinterleave=0',    'cinterleave=1']
        legend =     ['separate X and Z', 'XZ 1-column interleave']
        colors =     ['tab:green',        'tab:blue']
        style =      ['-',                '--']
        config2dir.clear()
        config2csv.clear()
        for make_config in make_configs:
            config_name = make_config.replace('=', '_c')
            run(config2dir, config2csv, make_config, 'default', config_name)
        print(" Finished %d circuits with %d con,figurations for %d times.%50s\n" %(len(qubits_range), len(make_configs), nsamples, " ")), sys.stdout.flush()
        collect(data, config2csv)
        plt.figure(figsize=(25,13))
        plt.rcParams["font.family"] = "Times New Roman"
        for i, config in enumerate(data.keys()):
            plt.plot(data[config][0],  data[config][1], linestyle=style[i], color=colors[i], linewidth=lW)
        show_plot(main_logs_dir + '/column_interleaving_plot', legend)
                
    if args.cpmix:
        print("Experimenting with column-interleaving and different {H,S}-gate frequencies..")
        make_configs=['cinterleave=0', 'cinterleave=1']
        configs = ['--H=0.125 --S=0.125', '--H=0.25 --S=0.25', '--H=0.375 --S=0.375']
        legend =  ['separate (H,S = %25)', 'separate (H,S = %50)', 'separate (H,S = %75)', 'interleave (H,S = %25)', 'interleave (H,S = %50)', 'interleave (H,S = %75)']
        colors =  ['tab:red',              'tab:blue',             'tab:green',            'tab:red',                'tab:blue',               'tab:green']
        style =   ['-',                    '-',                    '-',                    '--',                     '--',                     '--']
        config2dir.clear()
        config2csv.clear()
        for make_config in make_configs:
            for config in configs:
                config_name = make_config.replace('=', '_c') + '_' + config.replace('--', '').replace(' ', '_').replace('H=', 'H').replace('S=', 'S')
                run(config2dir, config2csv, make_config, config, config_name)
        print(" Finished %d circuits with %d configurations for %d times.%70s\n" %(len(qubits_range), len(make_configs) * len(configs), nsamples, " ")), sys.stdout.flush()
        collect(data, config2csv)
        plt.figure(figsize=(25,13))
        plt.rcParams["font.family"] = "Times New Roman"
        for i, config in enumerate(data.keys()):
            plt.plot(data[config][0],  data[config][1], linestyle=style[i], color=colors[i], linewidth=lW)
        show_plot(main_logs_dir + '/column_interleaving_HS_plot', legend)

benchmark()
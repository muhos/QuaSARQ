import os, sys
import argparse
import lzma
import subprocess
import csv
import random

Path = os.path

parent_dir = Path.dirname(Path.abspath(__file__))
main_dir = Path.dirname(parent_dir) + '/'

print("\t\t Equivalence Checking with QuaSARQ\n")

parser = argparse.ArgumentParser("gpu_benchmark")
parser.add_argument('-src', '--src', help="Directory to source code.", default='src')
parser.add_argument('-c', '--circuit', help="Directory to read circuts.", default='test')
parser.add_argument('-o', '--output', help="Directory to store results.", default='results/equivalence/quasarq')
parser.add_argument('-g', '--kernelconfig', help="Path of kernel configuration", default='src/kernel.config')
parser.add_argument('-n', '--nsamples', help="Number of samples", default=2)
parser.add_argument('-m', '--max', help="Maximum qubits", default=0)
args = parser.parse_args()

circuit_dir = main_dir + args.circuit
code_dir = main_dir + args.src
main_logs_dir = main_dir + args.output
kernelconfig = main_dir + args.kernelconfig
nsamples = int(args.nsamples)
max_qubits = int(args.max)

os.makedirs(main_logs_dir, exist_ok=True)

verbal = ['Circuits check', 'Failed state']
header = ['Circuit', 'Initial time', 'Schedule time', 'Simulation time', 'Energy consumption',
          'Circuits check', 'Failed state', 
          'Tableau partitions', 'Tableau memory', 'Tableau step speed', 
          'Circuit memory', 'Average parallel gates', 'Clifford gates']

random_line_idx = -1
random_gate = ''
org_gate = ''

def write_log_file(output, log_file):
    with open(log_file, 'w') as log_f:
        random_output = ''
        random_output += 'Random line: ' + str(random_line_idx) + '\n'
        random_output += 'Random gate: ' + random_gate + '\n'
        random_output += 'Original gate: ' + org_gate + '\n'
        log_f.write(random_output + str(output))
        log_f.close()

def compile_clean():
    args = ('make', '-C', code_dir, 'clean')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    args = ('make', '-C', code_dir, 'nocolor=1')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

def average_results(sample, in_cell, in_entry, run):
    for i, cell in enumerate(header):
        if (in_cell in verbal and cell == in_cell):
            sample[i] = in_entry
            break
        if (cell == in_cell):
            if ('time' in in_cell or 'speed' in in_cell or 'consumption'):
                sample[i] += float(in_entry)
                if (run == nsamples - 1):
                    sample[i] /= nsamples
            else:
                sample[i] = float(in_entry)
            break

def run_config(circuit, csvfile, benchpath, other_benchpath):
    verbose_opt = '--verbose=0'
    kernelconfig_opt = '--config-path=' + kernelconfig
    binaryfile = code_dir + '/quasarq'
    args = (binaryfile, benchpath, other_benchpath, '-equivalence', verbose_opt, kernelconfig_opt)
    row = [circuit]
    avg = [0] * len(header)
    for i in range(0, nsamples):
        print("QuaSARQ checking equivalence of [%-12s] the %d-time... " %(circuit, i+1), end='\r'), sys.stdout.flush()
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        log_file = main_logs_dir + '/' + 'log_run_' + str(i) + '_' + circuit + '.txt'
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
    csv_writer = csv.writer(csvfile)
    for i in range(1, len(header)):
        if not isinstance(avg[i], str):
            row.append("%.2f" % avg[i])
        else:
            row.append(avg[i])    
    csv_writer.writerow(row)

gate_1 = ['H', 'S_DAG', 'S', 'X', 'Y', 'Z']
gate_2 = ['CX', 'CY', 'CZ', 'ISWAP', 'SWAP']

def inject(lines):
    global random_line_idx, random_gate, org_gate
    other = list()
    random.seed(len(lines))
    random_line_idx = random.randint(0, len(lines))
    random_line = lines[random_line_idx]
    random_gate = ''
    for g in gate_2:
        if g in random_line:
            org_gate = g
            random_gate = gate_2[random.randint(0, len(gate_2) - 1)]
            break
    if random_gate == '':
        for g in gate_1:
            if g in random_line:
                org_gate = g
                random_gate = gate_1[random.randint(0, len(gate_1) - 1)]
                break
    assert(org_gate != '')
    assert(random_gate != '')
    random_line = random_line.replace(org_gate, random_gate)
    other = lines[:]
    other[random_line_idx] = random_line
    assert(random_gate in other[random_line_idx])
    return other

if __name__ == '__main__':
    print("Compiling directory %s from scratch... " %(code_dir), end=''), sys.stdout.flush()
    compile_clean()
    print("done.\n")

    csv_path = main_logs_dir + '/equivalence.csv'
    csv_f = open(csv_path, 'a', encoding='UTF8', newline='')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(header)

    file_list = [f for f in os.listdir(circuit_dir) if f.endswith(".xz")]
    file_list.sort()
    for i,file_name in enumerate(file_list):
        path = os.path.join(circuit_dir, file_name)
        with lzma.open(path, mode='rt', encoding='utf-8') as f:
            stim_string_file_name = os.path.splitext(file_name)
            file_name_stim = os.path.splitext(stim_string_file_name[0])
            written_file_path = file_name_stim[0] + '_tmp' + file_name_stim[1]
            other_written_file_path = file_name_stim[0] + '_other_tmp' + file_name_stim[1]
            if not os.path.exists(written_file_path) or not os.path.exists(other_written_file_path):
                lines = list()
                for line in f:
                    if '#' in line:
                        qubits = int(line.replace('#', ''))
                        if (max_qubits and qubits > max_qubits):
                            written_file_path = ''
                            break
                    lines.append(line)
                if (written_file_path == ''): continue              
                with open(written_file_path, 'w') as str_f: 
                    str_f.write(''.join(lines))
                str_f.close()
                other = inject(lines)
                with open(other_written_file_path, 'w') as str_f:
                    str_f.write(''.join(other))
                str_f.close()
            run_config(file_name_stim[0], csv_f, written_file_path, other_written_file_path)
            os.remove(written_file_path)
            os.remove(other_written_file_path)
    csv_f.close()
    print("QuaSARQ checked equivalence of %d circuits for %d times.%40s\n" %(len(file_list), nsamples, " ")), sys.stdout.flush()

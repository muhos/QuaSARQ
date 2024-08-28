import os, sys, csv
import subprocess
import multiprocessing
import argparse
import lzma
import stim
import random
import time

Path = os.path

parent_dir = Path.dirname(Path.abspath(__file__))
main_dir = Path.dirname(parent_dir) + '/'

parser = argparse.ArgumentParser("qiskit_benchmark")
parser.add_argument('-s', '--script', help="Script to measure power consumption (CPU-specific).", default='amd-power.py')
parser.add_argument('-c', '--circuit', help="Directory to read circuits.", default='test')
parser.add_argument('-o', '--output', help="Directory to store results.", default='results/equivalence/stim')
parser.add_argument('-n', '--nsamples', help="Number of samples", default=2)

args = parser.parse_args()
circuit_dir = main_dir + args.circuit
power_script = parent_dir + '/' + args.script
main_logs_dir = main_dir + args.output
nsamples = int(args.nsamples)

os.makedirs(main_logs_dir, exist_ok=True)

header = ['Circuit', 'Simulation time', 'Energy consumption',
          'Circuits check', 'Failed state']

power_draw = 70 # defualt over a range of modern CPUs.
circuit = None
other_circuit = None
plus_circuit = None
plus_other_circuit = None
circuitname = ''

random_line_idx = -1
random_gate = ''
org_gate = ''

qubits = 0

def write_log_file(output, log_file, mode):
    with open(log_file, mode) as log_f:
        random_output = ''
        if mode == 'w':
            random_output += 'Random line: ' + str(random_line_idx) + '\n'
            random_output += 'Random gate: ' + random_gate + '\n'
            random_output += 'Original gate: ' + org_gate + '\n'
        log_f.write(random_output + str(output))
        log_f.close()

def remove_signs(stabs):
    no_sign_stabs = []
    for gen in stabs:
        string = ''
        string += str(gen)
        no_sign_stabs.append(string[1:])
    return ''.join(no_sign_stabs)

def is_equal(stab1, stab2, sign = True):
    if (sign):
        if stab1 == stab2:
            return True
        return False
    else:
        assert(len(stab1) == len(stab2))
        n = len(stab1)
        for i in range(0, n):
            s1 = str(stab1[i])[1:]
            s2 = str(stab2[i])[1:]
            if (s1 != s2):
                return False
        return True

def run_config(id, returned):
    log_file = main_logs_dir + '/' + 'log_' + circuitname + '.txt'
    simtime = 0.0
    equivalence_check = 'NOT EQUIVALENT'
    failed_state = 'None'
    write_log_file('', log_file, 'w')
    row = [0] * len(header)
    row[0] = circuitname
    for i in range(0, nsamples):
        print(" CCEC checking equivalence of [%-12s] the %d-time... " %(circuitname, i+1), end='\r'), sys.stdout.flush()
        output = 'Run ' + str(i) + ': '
        start_time = time.monotonic()
        U = stim.TableauSimulator()
        U.do(circuit)
        V = stim.TableauSimulator()
        V.do(other_circuit)
        stabilizers1 = U.canonical_stabilizers()
        if len(stabilizers1) == 0:
            raise Exception('stabilizers1 size is 0')
        stabilizers2 = V.canonical_stabilizers()
        if len(stabilizers2) == 0:
            raise Exception('stabilizers2 size is 0')
        if is_equal(stabilizers1, stabilizers2):
            U = stim.TableauSimulator()
            U.do(plus_circuit)
            V = stim.TableauSimulator()
            V.do(plus_other_circuit)
            stabilizers1 = U.canonical_stabilizers()
            if len(stabilizers1) == 0:
                raise Exception('stabilizers1 size is 0')
            stabilizers2 = V.canonical_stabilizers()
            if len(stabilizers2) == 0:
                raise Exception('stabilizers1 size is 0')
            if is_equal(stabilizers1, stabilizers2):
                equivalence_check = 'EQUIVALENT'
            else:
                failed_state = '+'
        else:
            failed_state = '0'
        end_time = time.monotonic()
        elapsed_time = (end_time - start_time)
        output += str(elapsed_time) + '\n' 
        write_log_file(output, log_file, 'a')
        simtime += elapsed_time
    simtime /= nsamples
    row[1] = simtime
    row[-1] = failed_state
    row[-2] = equivalence_check
    returned[id] = row
    rounded = "%.2f" % simtime
    output = 'Simulation time: ' + rounded + ' seconds\n'
    output += '' + equivalence_check + '\n'
    output += failed_state + ' state\n'
    write_log_file(output, log_file, 'a')

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

def measure(id, returned):
    binaryfile = ['python3', power_script, '-f', '10', '-i', '100']
    popen = subprocess.Popen(binaryfile, stdout=subprocess.PIPE, stderr=subprocess.PIPE, user='root')
    popen.wait()
    output = ''
    output = popen.stdout.read().decode('utf-8')
    output += popen.stderr.read().decode('utf-8')
    returned[id] = output

if __name__ == '__main__':

    csv_path = main_logs_dir + '/equivalence.csv'
    csv_f = open(csv_path, 'a', encoding='UTF8', newline='')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(header)

    file_list = [f for f in os.listdir(circuit_dir) if f.endswith(".xz")]
    file_list.sort()
    for i,file_name in enumerate(file_list):
        circuit_path = os.path.join(circuit_dir, file_name)
        with lzma.open(circuit_path, mode='rt', encoding='utf-8') as f:
            # Swap a line and write a new circuit.
            lines = list()
            for line in f:
                if '#' in line:
                    qubits = int(line.replace('#', ''))
                lines.append(line)
            other = inject(lines)
            assert(other is not lines)
            string_file_name = os.path.splitext(os.path.basename(circuit_path))
            circuitname = os.path.splitext(string_file_name[0])[0]
            circuit = stim.Circuit(''.join(lines))
            if len(lines) != len(other):
                raise Exception('inconsistent circuits')
            other_circuit = stim.Circuit(''.join(other))
            if len(other_circuit) == 0:
                raise Exception('other circuit cannot be empty')
            # state +
            H_string = 'H'
            for q in range(0, qubits):
                H_string += ' ' + str(q)
            plus_lines = list()
            plus_lines.append(H_string)
            assert(len(lines) > 1)
            for line in lines:
                plus_lines.append(line)
            plus_circuit = stim.Circuit(''.join(plus_lines))
            if len(plus_circuit) == 0:
                raise Exception('plus circuit cannot be empty')
            plus_lines = list()
            plus_lines.append(H_string)
            for line in other:
                plus_lines.append(line)
            plus_other_circuit = stim.Circuit(''.join(plus_lines))
            if len(plus_other_circuit) == 0:
                raise Exception('plus circuit cannot be empty')
            plus_lines = list()
            lines = list()
            other = list()
        manager = multiprocessing.Manager()
        returned = manager.dict()
        p1 = multiprocessing.Process(target=run_config, args=(1, returned))
        p2 = multiprocessing.Process(target=measure, args=(2, returned))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        row = []
        for entry in returned[1]:
            row.append(entry)
        elapsed = returned[1][1]
        splitlines = returned[2].splitlines()
        if (len(splitlines) >= 2):
            power_draw = float(splitlines[1].split(':')[1])
        rounded = "%.2f" % (power_draw * elapsed)
        print(rounded)
        row[2] = rounded
        energyline = rounded + ' joules\n'
        log_file = main_logs_dir + '/' + 'log_' + circuitname + '.txt'
        write_log_file(energyline, log_file, 'a')
        csv_writer.writerow(row)
    csv_f.close()
        
    print(" CCEC checked equivalence of %d circuits for %d times.%40s\n" %(len(file_list), nsamples, " ")), sys.stdout.flush()

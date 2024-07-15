from timeit import default_timer as timer
import os, sys, csv
import argparse
import lzma
import qiskit.quantum_info 
import qiskit.qasm2

print("\t\t Qiskit Clifford Simulation\n")

parser = argparse.ArgumentParser("qiskit_benchmark")
parser.add_argument('-p', '--path', help="Benchmark file.", default='')
parser.add_argument('-r', '--resultsdir', help="Directory to store results.", default='results')
parser.add_argument('-n', '--nsamples', help="Number of samples", default=2)

args = parser.parse_args()
main_logs_dir = args.resultsdir
nsamples = int(args.nsamples)

path = args.path

header = ['Circuit', 'time']

csv_path = main_logs_dir + '/qiskit.csv'
csv_file = open(csv_path, 'a', encoding='UTF8', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(header)

def write_log_file(output, log_file):
    with open(log_file, 'w') as log_f:
        log_f.write(str(output))
        log_f.close()

def run_config(circuit, circuitname):
    row = [circuitname]
    simtime = 0
    qc = qiskit.qasm2.loads(circuit)
    #print(qc)
    output = ''
    for i in range(0, nsamples):
        output += 'Run ' + str(i) + 'time: '
        start = timer()
        qiskit.quantum_info.Clifford(qc)
        stop = timer()
        time = (stop - start)
        output += str(time) + ' seconds\n' 
        simtime += (stop - start)
    simtime /= nsamples
    output = 'Average time: ' + str(simtime) + ' seconds\n'
    log_file = main_logs_dir + '/' + 'log_' + circuitname + '.txt'
    write_log_file(output, log_file)
    rounded = "%.2f" % simtime
    row.append(rounded)
    csv_writer.writerow(row)

with lzma.open(path, mode='rt', encoding='utf-8') as f:
    print("Reading circuit file %s..." %(path), end=''), sys.stdout.flush()
    qasm_string_file_name = os.path.splitext(path)
    file_name_qasm = os.path.splitext(qasm_string_file_name[0])
    lines = list()
    for line in f:
        lines.append(line)
    qasm = ''.join(lines)
    run_config(qasm, file_name_qasm[0])  

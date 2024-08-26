from timeit import default_timer as timer
import os, sys, csv
import argparse
import lzma
import stim
import logging

from energy_consumption_reporter.energy_tester import EnergyTester
from energy_consumption_reporter.energy_model import EnergyModel
from energy_consumption_reporter.energy_tester import EnergyTester, OutputType

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

print("\t\t Stim Clifford Simulation\n")

parser = argparse.ArgumentParser("qiskit_benchmark")
parser.add_argument('-p', '--path', help="Benchmark file.", default='')
parser.add_argument('-r', '--output', help="Directory to store results.", default='results')
parser.add_argument('-n', '--nsamples', help="Number of samples", default=2)

args = parser.parse_args()
main_logs_dir = args.output
nsamples = int(args.nsamples)

path = args.path

circuit = None
circuitname = ''

def write_log_file(output, log_file, mode):
    with open(log_file, mode) as log_f:
        log_f.write(str(output))
        log_f.close()

@EnergyTester.energy_test(1)
def measure_power():
    def run_config():
        log_file = main_logs_dir + '/' + 'log_' + circuitname + '.txt'
        row = [circuitname]
        simtime = 0
        #print(qc)
        write_log_file('', log_file, 'w')
        for i in range(0, nsamples):
            output = 'Run ' + str(i) + ': '
            start = timer()
            t = stim.TableauSimulator().do_circuit(circuit)
            stop = timer()
            time = (stop - start)
            output += str(time) + '\n' 
            write_log_file(output, log_file, 'a')
            simtime += (stop - start)
        simtime /= nsamples
        rounded = "%.2f" % simtime
        output = rounded + ' seconds\n'
        write_log_file(output, log_file, 'a')
        return True
    assert run_config() == True

with lzma.open(path, mode='rt', encoding='utf-8') as f:
    #next(f)
    string_file_name = os.path.splitext(os.path.basename(path))
    circuitname = os.path.splitext(string_file_name[0])[0]
    circuit = stim.Circuit.from_file(f)
    json_file = main_logs_dir + '/' + 'energy_' + circuitname + '.json'
    EnergyTester().set_report_path(json_file)
    EnergyTester().set_save_report(OutputType.JSON)
    EnergyTester().set_zero_offset(False)
    measure_power()
    import json
    data = None
    with open(json_file, "r") as jf:
        data = json.load(jf)
        jf.close()
    power = data['results']['cases'][0]['power'][0]
    energy = data['results']['cases'][0]['energy'][0]
    log_file = main_logs_dir + '/' + 'log_' + circuitname + '.txt'
    write_log_file(str(power) + ' watt' + '\n' + str(energy/nsamples) + ' joules\n', log_file, 'a')


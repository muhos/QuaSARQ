from timeit import default_timer as timer
import os, sys
import argparse
import lzma
import stim

print("\t\t Stim Clifford Simulation\n")

parser = argparse.ArgumentParser("stim_benchmark")
parser.add_argument('-p', '--path', help="Directory to read benchmarks.", default='clifford-benchmarks/stim')
args = parser.parse_args()

bench_dir = args.path
file_list = [f for f in os.listdir(bench_dir) if f.endswith(".xz")]

for file_name in file_list:
    path = os.path.join(bench_dir, file_name)
    with lzma.open(path, mode='rt', encoding='utf-8') as f:
        print("Reading circuit file %s..." %(file_name), end=''), sys.stdout.flush()
        start = timer()
        next(f)
        c = stim.Circuit.from_file(f)
        stop = timer()
        readtime = stop - start
        print(" done in %.3f seconds" %(readtime))
        print(" Simulating %s..." %(file_name), end=''), sys.stdout.flush()
        start = timer()
        s=stim.TableauSimulator()
        s.do_circuit(c)
        #t = stim.Tableau.from_circuit(c)
        t = s.current_inverse_tableau()
        stop = timer()
        simtime = stop - start
        print(" done in %.4f seconds" %(simtime))
        print(" Time: %.4f seconds" %(readtime+simtime)), sys.stdout.flush()
        #print(c.diagram())
    
    
    

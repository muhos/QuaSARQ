#!/bin/bash

min_qubits=$1 
max_qubits=$2 
source_dir=$3
dest_dir=$4

mkdir -p $dest_dir

for c in $source_dir/*.xz
do
    qubits=${c##*/}
    qubits=${qubits%%.*}
    qubits=${qubits/q/''}
    qubits=${qubits/_d100/''}
    if (( $qubits >= $min_qubits )) && (( $qubits <= $max_qubits )); then
        echo -n "Moving $c to $dest_dir.."
        mv $c $dest_dir
        echo "done."
    fi
done
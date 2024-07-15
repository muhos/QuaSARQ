#!/bin/bash

resultdir=$1

[ ! -d $resultdir ] && echo "error: results directory '$resultdir' not found" && exit

csv_name=$2

[ -z "$csv_name" ] && echo "error: missing csv file name" && exit

csv_file=$resultdir/${csv_name}.csv

echo "Circuit, Time (sec), Energy (joules), Circuits Check, Failed state" > $csv_file

for f in $resultdir/*.txt
do
    circuit=${f##*/}
    circuit=${circuit%.*}
    timeline=$(grep 'seconds' $f)
    energyline=$(grep 'joules' $f)
    time=0
    energy=0
    check="EQUIVALENT"
    state="None"
	if [ ! -z "$timeline" ]; then 
		time=$(echo $timeline | awk '{ print $1 }')
		if (( $(echo "$time < 0.001" | bc -l) )); then
			time=$(echo "0.001" | bc -l)
		fi
        check=$(grep 'EQUIVALENT' $f)
        state=$(tail -n +9 $f)
    else
        timeline=$(grep 'Run 0' $f)
        if [ ! -z "$timeline" ]; then 
            time=$(echo $timeline | awk '{ print $3 }')
            if (( $(echo "$time < 0.001" | bc -l) )); then
                time=$(echo "0.001" | bc -l)
            fi
        fi
	fi
    if [ ! -z "$energyline" ]; then 
		energy=$(echo $energyline | awk '{ print $1 }')
		if (( $(echo "$energy < 0.001" | bc -l) )); then
			powerline=$(grep 'watt' $f)
            power=0
            if [ ! -z "$powerline" ]; then 
                power=$(echo $powerline | awk '{ print $1 }')
                energy=$(echo "($power - 5) * $time" | bc -l)
            fi
        fi
	fi
    echo "$circuit,$time,$energy,$check,$state" >> $csv_file
    
done
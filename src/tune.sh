#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path to circuits>"
  exit 1
fi

circuits="$1"

if [ ! -d "$circuits" ]; then
  echo "circuits not found!"
  exit 1
fi

log=tuning_log.txt

echo -n "" > $log

for f in $circuits/*.xz
do
  circuit=${f%%.xz}
  echo -n "uncompressing $f.."
  xz -dkq $f
  echo "done."
  ./quasarq $circuit -tune-all >> tuning_log.txt 2>&1
done
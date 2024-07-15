#!/bin/bash

outFile=sanitize.out

echo -n "Checking memory... "
compute-sanitizer --tool initcheck --track-unused-memory yes $@ > $outFile 2>&1
compute-sanitizer --tool memcheck --leak-check full --padding 32 --check-cache-control yes $@ >> $outFile 2>&1
echo "done."

echo -n "Checking data racing... "
compute-sanitizer --tool racecheck --racecheck-report all $@ >> $outFile 2>&1
echo "done."


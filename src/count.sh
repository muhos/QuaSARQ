#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <string> <file>"
  exit 1
fi

search_string="$1"
file="$2"

# Ensure file exists
if [ ! -f "$file" ]; then
  echo "File not found!"
  exit 1
fi

# Initialize counter
count=0

# Read the file line by line
while IFS= read -r line; do
  # Check if the line exactly matches the search string
  if [[ "$line" == *"$search_string"* ]]; then
    ((count++))
  fi
done < "$file"

# Output the result
echo "The string '$search_string' occurs $count times in the file '$file'."
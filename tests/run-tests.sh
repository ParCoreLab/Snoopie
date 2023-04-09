#!/usr/bin/env bash


echo "Running remote direct access tests:"
echo -n "  "

(
 set -e

 cd diim/ 
 (
   make clean
   make 
 ) > /dev/null

 exp_size=64

 # cut the csv header and count the number of operations
 recv_size=$(SIZE=$exp_size make run | sed '1d' | wc -l)

 # count how many mem_dev_ids was for device 1
 target_one=$(SIZE=$exp_size make run | sed '1d' | cut -f 5 -d ',' | grep -c '1')

 # count how many mem_dev_ids was for device 2
 target_two=$(SIZE=$exp_size make run | sed '1d' | cut -f 5 -d ',' | grep -c '2')

 if (( recv_size != exp_size )); then
   echo "DIIM failed. Expected ${exp_size} direct access operations. Recieved ${recv_size}."
   exit 1
 fi

 if (( exp_size / 2 != target_one )); then
   echo "DIIM failed. Expected $((exp_size / 2)) direct access operations on device one. Recieved ${target_one}."
   exit 1
 fi

 if (( exp_size / 2 != target_two )); then
   echo "DIIM failed. Expected $((exp_size / 2)) direct access operations on device two. Recieved ${target_two}."
   exit 1
 fi

 echo "DIIM is successful (size: $recv_size, target_one: $target_one, target_two: $target_two)"
)

#!/usr/bin/env bash

function run_diim() {
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
}


function run_nvshmem() {
  echo "Running nvshmem remote direct access tests:"
  echo -n " "

  (
   set -e 
   cd nvshmem/
   (
     make clean
     make
   ) > /dev/null
    
   np=4
   exp_size=64

   results=$(NP=$np SIZE=$exp_size make run)

   # cut the csv header and count the number of operations
   recv_size=$(echo "$results" | sed "1,${np}d" | wc -l)

   # check if the shift is working properly
   target_1=$(echo "$results" | sed "1,${np}d" | cut -f 5 -d ',' | grep -c '1')
   target_2=$(echo "$results" | sed "1,${np}d" | cut -f 5 -d ',' | grep -c '2')
   target_3=$(echo "$results" | sed "1,${np}d" | cut -f 5 -d ',' | grep -c '3')
   target_4=$(echo "$results" | sed "1,${np}d" | cut -f 5 -d ',' | grep -c '0')

   
   if (( exp_size != recv_size )); then
     echo "NVSHMEM failed. Expected ${exp_size} direct access operations. Recieved ${recv_size}."
     exit 1
   fi

   if (( target_1 != (exp_size / np) )); then
     echo "NVSHMEM failed. Expected $((exp_size / np)) direct access operations on device one. Recieved ${target_1}."
     exit 1
   fi

   if (( target_2 != (exp_size / np) )); then
     echo "NVSHMEM failed. Expected $((exp_size / np)) direct access operations on device two. Recieved ${target_2}."
     exit 1
   fi

   if (( target_3 != (exp_size / np) )); then
     echo "NVSHMEM failed. Expected $((exp_size / np)) direct access operations on device three. Recieved ${target_3}."
     exit 1
   fi

   if (( target_4 != (exp_size / np) )); then
     echo "NVSHMEM failed. Expected $((exp_size / np)) direct access operations on device four. Recieved ${target_4}."
     exit 1
   fi

   echo "NVSHMEM is successful (size: $recv_size, target_one: $target_1, target_two: $target_2, target_3: $target_3, target_4: $target_4)"
  )
}

function run_hipa() {
  echo "Running host initiated peer memcpy tests:"
  echo -n "  "
  (
   set -e

   cd hipa/ 
   (
     make clean
     make 
   ) > /dev/null

   exp_size=64

   # cut the csv header and count the number of operations
   recv_size=$(SIZE=$exp_size make run | sed '1d' | cut -f 10 -d ',')

   # get the sender id
   sender_id=$(SIZE=$exp_size make run | sed '1d' | cut -f 4 -d ',' | grep -c '0')

   # get the receiver id
   recv_id=$(SIZE=$exp_size make run | sed '1d' | cut -f 5 -d ',' | grep -c '1')

   if (( exp_size * 4 != recv_size )); then
     echo "HIPA failed. Expected ${exp_size} direct access operations. Recieved ${recv_size}."
     exit 1
   fi

   # cut the csv header and count the number of operations
   recv_size=$(SIZE=$exp_size make run_async | sed '1d' | cut -f 10 -d ',')

   # get the sender id
   sender_id=$(SIZE=$exp_size make run_async | sed '1d' | cut -f 4 -d ',' | grep -c '0')

   # get the receiver id
   recv_id=$(SIZE=$exp_size make run_async | sed '1d' | cut -f 5 -d ',' | grep -c '1')

   if (( exp_size * 4 != recv_size )); then
     echo "HIPA failed. Expected ${exp_size} direct access operations. Recieved ${recv_size}."
     exit 1
   fi

   echo "HIPA is successful (size: $recv_size)"
  )
}

run_diim
run_nvshmem
run_hipa

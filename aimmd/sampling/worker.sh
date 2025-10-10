#!/bin/bash

worker_id=$1
mdrun=$2

# kill gmx if worker is killed
cleanup() {
    if [[ -n "$pid" ]]; then
        pkill -TERM -P "$pid" 2>/dev/null
        kill -TERM "$pid" 2>/dev/null
        sleep 5
        pkill -KILL -P "$pid" 2>/dev/null
        kill -KILL "$pid" 2>/dev/null
    fi
    exit 0
}

# trap common termination signals
trap cleanup SIGINT SIGTERM SIGHUP EXIT

while true; do

  if [ -f "${worker_id}" ]; then
    # look for running file name in "<id>.mdrun"
    fname=$(<"${worker_id}")  # specific to id
      
    $mdrun -deffnm $fname -cpo ${fname}.cpt -cpi ${fname}.cpt -cpt .1 -nobackup &
    pid=$!
    
    # monitor loop: kill mdrun if "worker_id" is not present anymore
    while kill -0 $pid 2>/dev/null; do
      if [ ! -f "${worker_id}" ]; then
        kill -TERM $pid
        break
      fi
      sleep .25  # TODO check different times once again
      done
        
    wait $pid
  fi
done

# NOT WORKING NOW how to cancel as soon as problem??
# but only problem on exectution
#exit_code=$?
#if [ $exit_code -ne 0 ]; then
#    scancel $jobid
#fi

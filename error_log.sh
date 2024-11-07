#!/bin/bash
julia --project scripts/iterative_refinement.jl
ret=$?
#
#  returns > 127 are a SIGNAL
#
if [ $ret -gt 127 ]; then
        sig=$((ret - 128))
        echo "Got SIGNAL $sig"
        if [ $sig -eq $(kill -l SIGKILL) ]; then
                echo "process was killed with SIGKILL"
                dmesg > $HOME/dmesg-kill.log
        fi
fi

# bash error_log.sh
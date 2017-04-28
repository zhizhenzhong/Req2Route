#!/bin/bash
while [ "1" = "1" ]
do
    # git pull
    ps aux | grep window | grep -v grep > ps.txt
    nvidia-smi >> ps.txt
    rm *.log
    cp ~/projects/pyres/res/routing/*.log .
    git add .
    git commit -m "auto check"
    git push
    sleep 1800 # 1800: 30min, 3600: 1h
done

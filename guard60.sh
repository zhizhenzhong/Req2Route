#!/bin/bash
while [ "1" = "1" ]
do
    # git pull
    ps aux | grep domain | grep -v grep > ps.txt
    nvidia-smi >> ps.txt
    rm *.json
    rm *.log
    cp ~/projects/pyres/res_domain/*.log .
    git add .
    git commit -m "auto check"
    git push
    sleep 3600 # 1800: 30min, 3600: 1h
done

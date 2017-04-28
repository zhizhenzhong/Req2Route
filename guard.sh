#!/bin/bash
while [ "1" = "1" ]
do
    # git pull
    ps aux | grep global_attention | grep -v grep > ps.txt
    nvidia-smi >> ps.txt
    rm *.json
    rm *.log
    cp ~/projects/pyres/res_attn/*.json .
    cp ~/projects/pyres/res_attn/*.log .
    git add .
    git commit -m "auto check"
    git push
    sleep 1800 # 1800: 30min, 3600: 1h
done

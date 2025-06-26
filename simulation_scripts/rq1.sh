#!/bin/bash

(
    environments=("S3R3" "S6R3" "1Dlh" "1Dlhb" "N2S4" "N4S5")
    subgoal_types=("relative" "representation" "language")
    seeds=(8191 65537 104729)

    gpu_list=(0 1)
    max_simultaneous=40
    sims_started=0

    for env in "${environments[@]}"; do
        for seed in "${seeds[@]}"; do
            for subgoal_type in "${subgoal_types[@]}"; do
                folder_name="${env}_1_${subgoal_type:0:3}_${seed}"
                log_file="logs/${folder_name}.log"
                gpu="${gpu_list[sims_started % ${#gpu_list[@]}]}"
                python3 -m scripts.train \
                    --env "$env" \
                    --seed "$seed" \
                    --use-subgoal \
                    --subgoal-type "$subgoal_type" \
                    --subgoal-reward-value 0.2 \
                    --use-gpu \
                    --gpu-id "$gpu" \
                    --model "final/rq1/$folder_name" \
                    --frames 15000000 \
                    > "$log_file" 2>&1 &
                
                ((sims_started++))
                
                if (( sims_started >= max_simultaneous )); then
                    wait
                    sims_started=0
                fi
            done
        done
    done
    wait
) &

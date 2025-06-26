#!/bin/bash

(
    environments=("S6R3")
    subgoal_types=("relative")
    seeds=(8191 65537 104729)

    gpu_list=(0)
    max_simultaneous=40
    sims_started=0

    for env in "${environments[@]}"; do
        for seed in "${seeds[@]}"; do
            for subgoal_type in "${subgoal_types[@]}"; do
                folder_name="${env}_1_${subgoal_type:0:3}_${seed}_5rsubgoals"
                log_file="logs/${folder_name}.log"
                gpu="${gpu_list[sims_started % ${#gpu_list[@]}]}"
                python3 -m scripts.train \
                    --env "$env" \
                    --eval-envs "$env" \
                    --eval-nrandom-subgoals 5 \
                    --seed "$seed" \
                    --use-subgoal \
                    --subgoal-type "$subgoal_type" \
                    --subgoal-accuracy 1 \
                    --subgoal-mean 0 \
                    --subgoal-std 0 \
                    --subgoal-reward-value 1 \
                    --nrandom-subgoals 5 \
                    --use-gpu \
                    --gpu-id "$gpu" \
                    --model "gen2/$folder_name" \
                    --frames 250000000 \
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

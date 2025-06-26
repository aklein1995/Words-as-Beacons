#!/bin/bash

(
    environments=("S3R3" "S6R3" "1Dlh" "1Dlhb" "N2S4" "N4S5")
    subgoal_types=("relative" "representation" "language")
    seeds=(8191 65537 104729)

    parameter_sets=()
    while IFS= read -r line; do
        parameter_sets+=("$line")
    done < ./simulation_scripts/rq3_config

    gpu_list=(0 2)
    max_simultaneous=40
    sims_started=0

    for env in "${environments[@]}"; do
        for seed in "${seeds[@]}"; do
            for subgoal_type in "${subgoal_types[@]}"; do
                for param_set in "${parameter_sets[@]}"; do
                    IFS=' ' read -r param_env param_subgoal_type llm accuracy mean std <<< "$param_set"

                    if [[ "$param_env" != "$env" || "$param_subgoal_type" != "$subgoal_type" ]]; then
                        continue
                    fi

                    accuracy_formatted=$(echo "$accuracy" | tr '.' '-')
                    folder_name="${env}_${llm}_${accuracy_formatted}_${subgoal_type:0:3}_${seed}"
                    log_file="logs/${folder_name}.log"
                    
                    gpu="${gpu_list[sims_started % ${#gpu_list[@]}]}"

                    python3 -m scripts.train \
                        --env "$env" \
                        --seed "$seed" \
                        --use-subgoal \
                        --subgoal-type "$subgoal_type" \
                        --use-gpu \
                        --gpu-id "$gpu" \
                        --model "rq3/$folder_name" \
                        --subgoal-accuracy "$accuracy" \
                        --subgoal-mean "$mean" \
                        --subgoal-std "$std" \
                        --subgoal-reward-value 0.2 \
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
    done

    wait
) &

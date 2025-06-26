#!/bin/bash

(
    environments=("S3R3" "S6R3" "1Dlh" "1Dlhb" "N2S4" "N4S5")
    seeds=(8191 65537 104729)

    parameter_sets=()
    while IFS= read -r line; do
        parameter_sets+=("$line")
    done < ./simulation_scripts/rq4_config

    gpu_list=(0 1)
    max_simultaneous=50
    sims_started=0

    for env in "${environments[@]}"; do
        for seed in "${seeds[@]}"; do
            for param_set in "${parameter_sets[@]}"; do
                IFS=' ' read -r param_env llm accuracy <<< "$param_set"

                if [[ "$param_env" != "$env" ]]; then
                    continue
                fi

                if [[ "$env" == "S6R3" ]]; then
                    frames=30000000
                else
                    frames=15000000
                fi

                accuracy_formatted=$(echo "$accuracy" | tr '.' '-')
                folder_name="${env}_${llm}_${accuracy_formatted}_${seed}_noreward"
                log_file="logs/${folder_name}.log"
                
                gpu="${gpu_list[sims_started % ${#gpu_list[@]}]}"

                python3 -m scripts.train \
                    --env "$env" \
                    --seed "$seed" \
                    --use-subgoal \
                    --subgoal-type "representation" \
                    --use-gpu \
                    --gpu-id "$gpu" \
                    --model "final/rq4/$folder_name" \
                    --subgoal-accuracy "$accuracy" \
                    --subgoal-reward-value 0 \
                    --frames "$frames" \
                    > "$log_file" 2>&1 &

                ((sims_started++))

                folder_name="${env}_${llm}_${accuracy_formatted}_${seed}_nosubgoal"
                log_file="logs/${folder_name}.log"

                gpu="${gpu_list[sims_started % ${#gpu_list[@]}]}"

                python3 -m scripts.train \
                    --env "$env" \
                    --seed "$seed" \
                    --use-gpu \
                    --gpu-id "$gpu" \
                    --model "final/rq4/$folder_name" \
                    --subgoal-accuracy "$accuracy" \
                    --subgoal-reward-value 0.2 \
                    --frames "$frames" \
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

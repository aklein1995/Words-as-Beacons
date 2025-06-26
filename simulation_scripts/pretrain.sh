#!/bin/bash

pretrain_frames_list=(100000 250000 500000 1000000 2500000)
subgoal_accuracy_list=(1 0.9323)

gpu=0
model_name_prefix="pretrain/S6R3_pretrain"
env="S6R3"
seed=8191
subgoal_reward_value=0.2

for subgoal_accuracy in "${subgoal_accuracy_list[@]}"; do
    for pretrain_frames in "${pretrain_frames_list[@]}"; do
        (
            if [ "$pretrain_frames" -ge 1000000 ]; then
                formatted_frames="$(($pretrain_frames / 1000000))M"
            else
                formatted_frames="$(($pretrain_frames / 1000))k"
            fi

            if [ "$subgoal_accuracy" == "1" ]; then
                accuracy_type="Oracle"
            else
                accuracy_type="Llama"
            fi

            model_name="${model_name_prefix}_${formatted_frames}_${accuracy_type}_lan_02"

            python3 -m scripts.train \
                --env "$env" \
                --seed "$seed" \
                --use-subgoal \
                --subgoal-type language \
                --use-gpu \
                --gpu-id "$gpu" \
                --model "$model_name" \
                --subgoal-accuracy "$subgoal_accuracy" \
                --subgoal-mean 0 \
                --subgoal-std 0 \
                --pretrain 1 \
                --subgoal-reward-value "$subgoal_reward_value" \
                --frames "$pretrain_frames" > "pretrain_log_${formatted_frames}_${accuracy_type}.txt" 2>&1 &

            wait

            echo "Starting final training step for pretrain_frames=${pretrain_frames}, subgoal_accuracy=${subgoal_accuracy}..."

            python3 -m scripts.train \
                --env "$env" \
                --seed "$seed" \
                --use-subgoal \
                --subgoal-type language \
                --use-gpu \
                --gpu-id "$gpu" \
                --model "$model_name" \
                --subgoal-accuracy "$subgoal_accuracy" \
                --subgoal-mean 0 \
                --subgoal-std 0 \
                --subgoal-reward-value "$subgoal_reward_value" \
                --frames 15000000 > "final_log_${formatted_frames}_${accuracy_type}.txt" 2>&1 &

            wait
            if [ $? -ne 0 ]; then
                echo "Error during final training step for pretrain_frames=${pretrain_frames}, subgoal_accuracy=${subgoal_accuracy}. Exiting."
                exit 1
            fi

            echo "All tasks completed successfully for pretrain_frames=${pretrain_frames}, subgoal_accuracy=${subgoal_accuracy}."
        ) &
    done
done

wait

echo "All simulations completed successfully."

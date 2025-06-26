# Words as Beacons: Guiding RL Agents with High-Level Language Prompts

**[Paper](https://arxiv.org/abs/2410.08632)**

by Unai Ruiz-Gonzalez, Alain Andres, Javier Del Ser

Sparse reward environments in reinforcement learning (RL) pose significant challenges for exploration, often leading to inefficient or incomplete learning processes. To tackle this issue, this work proposes a teacher-student RL framework that leverages Large Language Models (LLMs) as ``teachers'' to guide the agent's learning process by decomposing complex tasks into subgoals. Due to their inherent capability to understand problems based on a textual description of structure and purpose, LLMs can provide subgoals to accomplish the task defined for the environment in a similar fashion to how a human would do. In doing so, three types of subgoals are proposed: positional targets relative to the agent, object representations and language-based instructions generated directly by the LLM. More importantly, we show that it is possible to query the LLM only during the training phase, enabling agents to operate during deployment without any LLM intervention. We assess the performance of this proposed framework by evaluating three state-of-the-art open source LLMs (Llama, DeepSeek, Qwen), allowing results to be replicated by anyone without cost barriers. Furthermore, we evaluate our approach across various procedurally generated environments of the MiniGrid benchmark, which require generalization to unseen task variations. Experimental results demonstrate that this teacher-student based approach accelerates learning and enhances exploration in complex tasks, achieving faster convergence in training steps compared to recent teacher-student setups designed for sparse reward environments.


## Table of Contents

- [Installation](#installation)
- [Directory layout](#layout)

## Layout

    .
    ├── gym_minigrid/                    # Minigrid environment 
    │   ├── envs
    │   │    └── subgoal_generator.py    # All logic to add subgoals to Minigrid
    │
    ├── plotting                         # Code to create the figures in the paper
    │
    ├── scripts                          # Scripts to train, evaluate, play, record and visualize episodes.
    │
    ├── simulation_scripts               # Files to launch all the experimentation
    │
    ├── subgoals                         # Folder with all the previously generated subgoals
    │                                    # and all the files to parse and clean subgoals.
    │
    ├── torch_ac                         # PPO implementation
    │
    ├── utils                            # Utility scripts
    │
    └── README.md

## Citation

If you further explore using our work, please cite it using the following bibtex.
```
@article{ruizgonzalez2024words,
    title=Words as Beacons: Guiding RL Agents with High-Level Language Prompts,
    author={Ruiz-Gonzalez, Unai and Andres, Alain and G.Bascoy, Pedro and Del Ser, Javier},
    year={2024},
    month={10},
    journal={arXiv preprint arXiv:2410.08632}
}
```
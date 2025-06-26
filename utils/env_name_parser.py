import gym_minigrid
import gym

ENV_DICT = {
    "S3R3": "MiniGrid-KeyCorridorS3R3-v0",
    "S4R3": "MiniGrid-KeyCorridorS4R3-v0",
    "S5R3": "MiniGrid-KeyCorridorS5R3-v0",
    "S6R3": "MiniGrid-KeyCorridorS6R3-v0",
    "1Dl": "MiniGrid-ObstructedMaze-1Dl-v0",
    "1Dlh": "MiniGrid-ObstructedMaze-1Dlh-v0",
    "1Dlhb": "MiniGrid-ObstructedMaze-1Dlhb-v0",
    "2Dl": "MiniGrid-ObstructedMaze-2Dl-v0",
    "N2S4": "MiniGrid-MultiRoom-N2-S4-v0",
    "N3S4": "MiniGrid-MultiRoom-N3-S4-v0",
    "N4S5": "MiniGrid-MultiRoom-N4-S5-v0",
    "E5": "MiniGrid-Empty-5x5-v0"
}

def parse_env_name(env_name):
    registered_envs = [spec.id for spec in gym.envs.registry.all()]
    
    if type(env_name) == str:
        if env_name in registered_envs:
            return env_name
    
        # Check if the acronym is passed
        try:
            return ENV_DICT[env_name]
        except KeyError:
            raise ValueError(f"No environment with name {env_name}. If the environment is a MiniGrid environment, please register within gym_minigrid/envs/__init__.py. If the environment is an acronym, please add it to the ENV_DICT dictionary in the utils/env_name_parser.py file.")
    elif type(env_name) == list:
        parsed_envs = []

        for env in env_name:
            if env_name in registered_envs:
                parsed_envs.append(env)
            else:
                try:
                    parsed_envs.append(ENV_DICT[env])
                except KeyError:
                    raise ValueError(f"No environment with name {env_name}. If the environment is a MiniGrid environment, please register within gym_minigrid/envs/__init__.py. If the environment is an acronym, please add it to the ENV_DICT dictionary in the utils/env_name_parser.py file.")
        return parsed_envs
import random
import numpy as np
import json
import pickle
import copy

def expand_or_match(value, target_length):

    if isinstance(value, list):
        assert len(value) in [1, target_length], \
            f"Accuracies, Means and STDs must be a float/int or a list of length 1 or {target_length}, but got a list of length {len(value)}"
    elif isinstance(value, float):
        assert isinstance(value, float), \
            f"Accuracies, Means and STDs must be a float/int or a list of length 1 or {target_length}, but got a {type(value).__name__}"
    else:
        assert isinstance(value, int), \
            f"Accuracies, Means and STDs must be a float/int or a list of length 1 or {target_length}, but got a {type(value).__name__}"

    if isinstance(value, list):
        return value if len(value) == target_length else value * target_length
    return [value] * target_length

class SubgoalGenerator:
    all_subgoals = None

    def __init__(self, subgoals, subgoal_file, subgoal_type, accuracy, mean, std, n_random_subgoals, pretrain, pretrain_subgoal_distance, env, seed, goal_pos, force_doors, include_walls, include_locked_room):
        self.subgoals = subgoals
        self.subgoal_file = subgoal_file
        self.subgoal_type = subgoal_type

        self.subgoal_accuracy = expand_or_match(accuracy, len(self.subgoals))
        self.subgoal_mean_error = expand_or_match(mean, len(self.subgoals))
        self.subgoal_std_error = expand_or_match(std, len(self.subgoals))

        self.n_random_subgoals = int(n_random_subgoals)

        self.pretrain = pretrain
        self.pretrain_subgoal_distance = pretrain_subgoal_distance

        # WARNING!: This code sets up a circular dependency, this can be problematic, modify at your own risk. 
        self.env = env
        self.seed = seed
        self.goal_pos = goal_pos

        self.force_doors = force_doors
        self.include_walls = include_walls
        self.include_locked_room = include_locked_room

        self.current_subgoal_index = 0
        self.subgoal_reward = True
        self.subgoal_steps = 0

        if self.subgoal_type == "language":
            with open("./subgoals/pool/data.pkl", 'rb') as f:
                self.language_pool = pickle.load(f)

        if self.subgoal_file:
            if not SubgoalGenerator.all_subgoals:
                self.get_subgoals_from_file()
            self.get_subgoals_by_seed()
        else:
            self.subgoals = self.replace_subgoals()

            if self.n_random_subgoals:
                if self.subgoal_type == "relative":
                    self.add_relative_random_subgoals()

                elif self.subgoal_type == "representation" or self.subgoal_type == "language":
                    self.add_representation_random_subgoals()

        self.completed_subgoals = [False for _ in range(len(self.subgoals))]
        
        try:
            self.current_subgoal = {
                "id": self.subgoals[self.current_subgoal_index][0],
                "subgoal": self.gen_current_subgoal(),
                "pos": self.subgoals[self.current_subgoal_index][1],
            }
        except IndexError:
            self.subgoal_reward = False
            self.current_subgoal = {
                "id": "no subgoal",
                "subgoal": self.fallback_subgoal(),
                "pos": (0, 0),
            }
        
    def get_subgoals_from_file(self):
        with open(self.subgoal_file, 'rb') as f:
            SubgoalGenerator.all_subgoals = json.load(f)

    def get_subgoals_by_seed(self):
        for entry in SubgoalGenerator.all_subgoals:
            if entry['seed'] == self.seed:
                if self.subgoal_type == "relative":
                    subgoal_pos = entry['relative']
                elif self.subgoal_type == "representation" or self.subgoal_type == "language":
                    subgoal_pos = entry['positions']
                break
    
        subgoal_pos = [pos for pos in subgoal_pos if pos is not None and pos != 'null']
        subgoal_pos = [pos for pos in subgoal_pos if pos[0] < self.env.width and pos[0] >= 0 and pos[1] < self.env.height and pos[1] >= 0]         
        self.subgoals = [(f"subgoal_{i}", pos) for i, pos in enumerate(subgoal_pos)]

    def replace_subgoals(self):
        if not self.pretrain:
            modified_subgoals = []
            for i, subgoal in enumerate(self.subgoals):
                accuracy = self.subgoal_accuracy[i]
                if random.random() >= accuracy:
                    if self.subgoal_type == "relative" or self.subgoal_type == "absolute":
                        pos = self.random_pos(subgoal[1], i)
                    elif self.subgoal_type == "representation" or self.subgoal_type == "language":
                        pos = self.random_object()
                    id = self.env.grid.get(*pos)
                    id = id.type if id else "floor"
                    subgoal = (id, pos)
                    
                modified_subgoals.append(subgoal)
            return modified_subgoals
        else: 
            if self.subgoal_type == "relative":
                subgoals = self.relative_pretrain()
            elif self.subgoal_type == "representation" or self.subgoal_type == "language":
                subgoals = self.representation_pretrain()
            return subgoals

    def add_relative_random_subgoals(self):
        for i in range(self.n_random_subgoals):
            for _ in range(10000):
                random_pos = (random.randint(1, self.env.width - 2), random.randint(1, self.env.height - 2))
                cell = self.env.grid.get(*random_pos)

                if self.env.room_from_pos(*self.goal_pos).pos_inside(*random_pos):
                    continue

                if cell and cell.type == "wall":
                    continue
                break

            random_subgoal = (f"random_subgoal_{i}", random_pos)
            self.subgoals.insert(random.randint(0, len(self.subgoals)-1), random_subgoal)

    def add_representation_random_subgoals(self):
        for i in range(self.n_random_subgoals):
            random_subgoal = (f"random_subgoal_{i}", self.random_object())
            self.subgoals.insert(random.randint(0, len(self.subgoals)-1), random_subgoal)

    def random_pos(self, pos, i):
        manhattan_distance = np.random.normal(self.subgoal_mean_error[i], self.subgoal_std_error[i])
        manhattan_distance = max(1, manhattan_distance)

        possible_subgoals = []
        for dx in range(int(manhattan_distance), int(manhattan_distance) + 1):
            dy = int(manhattan_distance) - abs(dx)
            possible_subgoals.append((dx, dy))
            if dy != 0:
                possible_subgoals.append((dx, -dy))
        random.shuffle(possible_subgoals)

        for dx, dy in possible_subgoals:
            new_x = pos[0] + dx
            new_y = pos[1] + dy
        
            if 0 <= new_x < self.env.width - 1 and 0 <= new_y < self.env.height - 1:
                return (new_x, new_y)
        
        return (random.randint(0, self.env.width-1), random.randint(1, self.env.height-1))

    def random_object(self):
        for _ in range(10000):
            random_object = (random.randint(1, self.env.width - 2), random.randint(1, self.env.height - 2))
            cell = self.env.grid.get(*random_object)

            if not self.include_locked_room and self.env.room_from_pos(*self.goal_pos).pos_inside(*random_object):
                continue

            if cell == None:
                continue

            if self.force_doors and cell.type != "door":
                continue

            if not self.include_walls and cell.type == "wall":
                continue
            break

        return random_object

    def relative_pretrain(self):
        possible_positions = []
        subgoals = []
        agent_x, agent_y = self.env.agent_pos

        for dx in range(-self.pretrain_subgoal_distance, self.pretrain_subgoal_distance+1):
            for dy in range(-self.pretrain_subgoal_distance, self.pretrain_subgoal_distance+1):
                if abs(dx) + abs(dy) == self.pretrain_subgoal_distance:
                    possible_positions.append((agent_x + dx, agent_y + dy))

        for subgoal in possible_positions:
            if subgoal[0] >= self.env.grid.height or subgoal[0] < 0 or subgoal[1] >= self.env.grid.width or subgoal[1] < 0:
                continue
            
            if not self.env.grid.get(*subgoal):
                subgoals.append(("subgoal", subgoal))
        
        subgoals = random.sample(subgoals, 1)
        return subgoals
    
    def representation_pretrain(self):
        assert "keycorridor" in self.env.__class__.__name__.lower(), \
            "Only 'KeyCorridor' environments are allowed for representation/language at this stage."
        agent_pos = self.env.agent_pos
        key_pos = self.subgoals[0][1]
        door_positions = self.env.doors_pos

        def manhanttan_distance(pos1, pos2):
            x_weight = 3
            y_weight = 1
            return x_weight * abs(pos1[0] - pos2[0]) + y_weight * abs(pos1[1] - pos2[1])
            # return abs(pos1[0] - pos2[0])

        closest_door = None
        min_distance = float('inf')

        for door in door_positions:
            distance = manhanttan_distance(agent_pos, door) #+ manhanttan_distance(key_pos, door)
            if distance < min_distance:
                min_distance = distance
                closest_door = door

        subgoal = [("subgoal", closest_door)]
        return subgoal

    def update_current_subgoal(self):
        if self.current_subgoal_index + 1 == len(self.subgoals):
            self.subgoal_reward = False
            self.current_subgoal = {
                "id": "no subgoal",
                "subgoal": self.fallback_subgoal(),
                "pos": (0, 0),
            }
            return

        try:
            self.current_subgoal = {
                "id": self.subgoals[self.current_subgoal_index][0],
                "subgoal": self.gen_current_subgoal(),
                "pos": self.subgoals[self.current_subgoal_index][1],
            }
        except IndexError:
            self.subgoal_reward = False
            self.current_subgoal = {
                "id": "no subgoal",
                "subgoal": self.fallback_subgoal(),
                "pos": (0, 0),
            }

    def gen_current_subgoal(self):
        subgoal_pos = self.subgoals[self.current_subgoal_index][1]

        if self.subgoal_type == "relative":
            subgoal = subgoal_pos - self.env.agent_pos
        elif self.subgoal_type == "absolute":
            subgoal = subgoal_pos
        elif self.subgoal_type == "representation":
            if self.env.grid.get(*subgoal_pos) == None:
                return self.fallback_subgoal()
            subgoal = self.env.grid.get(*subgoal_pos).encode()
        elif self.subgoal_type == "language":
            subgoal = self.env.grid.get(*subgoal_pos)
            nametype = subgoal.__class__.__name__.lower()

            if subgoal is None:
                return self.fallback_subgoal()
            if nametype == "door":
                if subgoal.state() == "locked":
                    subgoal = self.language_pool["locked_door"]
                else:
                    subgoal = self.language_pool[f"{subgoal.color}_door"]
            elif nametype == "key":
                subgoal = self.language_pool["key"]
            elif nametype == "box":
                subgoal = self.language_pool["box"]
            elif nametype == "ball":
                if "keycorridor" in self.env.__class__.__name__.lower():
                    subgoal = self.language_pool["ball"]
                else:
                    subgoal = self.language_pool[f"{subgoal.color}_ball"]
            elif nametype == "goal":
                subgoal = self.language_pool["goal"]
        else:
            raise ValueError(f"subgoal_type: {self.subgoal_type} is not valid...")
        
        return subgoal

    def check_subgoal_completion_before(self):
        self.subgoal_cells_before = []
        for id, subgoal_pos in self.subgoals:
            self.subgoal_cells_before.append(copy.deepcopy(self.env.grid.get(*subgoal_pos)))

    def check_subgoal_completion_after(self):
        self.subgoal_steps += 1
        info = {}
        info['subgoal_completed'] = False
        info['subgoal_steps'] = self.subgoal_steps

        for i, (id, subgoal_pos) in enumerate(self.subgoals):
            subgoal_cell_after = copy.deepcopy(self.env.grid.get(*subgoal_pos))

            if tuple(subgoal_pos) == tuple(self.env.agent_pos.tolist()) or self.subgoal_cells_before[i] != subgoal_cell_after:
                self.completed_subgoals[i] = True
                if i == self.current_subgoal_index:
                    info['subgoal_completed'] = True if self.subgoal_reward else False
                    self.increment_subgoal()
        
        info['current_subgoal_index'] = self.current_subgoal_index
        info['completed_subgoals'] = self.completed_subgoals

        info['subgoal'] = self.get_current_subgoal()
        info['subgoal_pos'] = self.get_current_subgoal_pos()
        info['subgoal_accuracy'] = self.subgoal_accuracy
        info['subgoal_mean'] = self.subgoal_mean_error
        info['subgoal_std'] = self.subgoal_std_error
        info['env_max_steps'] = self.env.max_steps

        return info

    def increment_subgoal(self):
        # If the last subgoal is not correct, there would not be subgoal until the end of the episode.
        if self.current_subgoal_index + 1 == len(self.subgoals):
            self.subgoal_reward = False
            self.current_subgoal = {
                "id": "no subgoal",
                "subgoal": self.fallback_subgoal(),
                "pos": (0, 0),
            }
            return

        self.current_subgoal_index += 1
        self.current_subgoal = {
            "id": self.subgoals[self.current_subgoal_index][0],
            "subgoal": self.get_current_subgoal(),
            "pos": self.subgoals[self.current_subgoal_index][1]
        }
        self.subgoal_steps = 0

    def get_current_subgoal(self):
        self.update_current_subgoal()
        return self.current_subgoal['subgoal']

    def get_current_subgoal_id(self):
        self.update_current_subgoal()
        return self.current_subgoal['id']
    
    def get_current_subgoal_pos(self):
        self.update_current_subgoal()
        return self.current_subgoal['pos']

    def fallback_subgoal(self):
        if self.subgoal_type == "relative" or self.subgoal_type == "absolute":
            return (0, 0)
        elif self.subgoal_type == "representation":
            return (-1, -1, -1)
        elif self.subgoal_type == "language":
            return np.zeros(384)
        else:
            raise ValueError(f"subgoal_type: {self.subgoal_type} is not valid...")
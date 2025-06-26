from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register

from .subgoal_generator import SubgoalGenerator

class KeyCorridor(RoomGrid):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        subgoal_file=None,
        subgoal_type="relative",
        subgoal_accuracy=[1],
        subgoal_mean_error=[0],
        subgoal_std_error=[0],
        n_random_subgoals=0,
        pretrain=False,
        pretrain_subgoal_distance=0,
        seed=None
    ):
        self.obj_type = obj_type
        self.subgoal_file = subgoal_file
        self.subgoal_type = subgoal_type
        self.subgoal_accuracy = subgoal_accuracy
        self.n_random_subgoals = n_random_subgoals

        self.subgoal_mean_error = subgoal_mean_error
        self.subgoal_std_error = subgoal_std_error

        self.pretrain = pretrain
        self.pretrain_subgoal_distance = pretrain_subgoal_distance

        self.episode_seed = seed

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30*room_size**2,
            seed=seed,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, locked_door_pos = self.add_door(2, room_idx, 2, locked=True)
        obj, obj_type_pos = self.add_object(2, room_idx, kind=self.obj_type)
        self.goal_pos = obj_type_pos

        # Add a key in a random room on the left side
        key, key_pos = self.add_object(0, self._rand_int(0, self.num_rows), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        doors, self.doors_pos = self.connect_all()

        # Subgoal generation
        subgoals = [
            ("key", list(key_pos)),
            ("locked_door", list(locked_door_pos)),
            ("ball", list(obj_type_pos))
        ]
        self.subgoal_generator = SubgoalGenerator(subgoals, subgoal_file=self.subgoal_file,
                                                  subgoal_type=self.subgoal_type,
                                                  accuracy=self.subgoal_accuracy,
                                                  mean=self.subgoal_mean_error,
                                                  std=self.subgoal_std_error,
                                                  n_random_subgoals=self.n_random_subgoals,
                                                  pretrain=self.pretrain,
                                                  pretrain_subgoal_distance=self.pretrain_subgoal_distance,
                                                  env=self, # Circular dependency, be careful modifying.
                                                  seed=self.episode_seed,
                                                  goal_pos=self.goal_pos,
                                                  force_doors=True,
                                                  include_walls=False,
                                                  include_locked_room=False)

        # Mission generation
        self.obj = obj
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def step(self, action):
        self.subgoal_generator.check_subgoal_completion_before()

        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.n_random_subgoals > 0:
                if self.carrying and self.carrying == self.obj and self.subgoal_generator.current_subgoal_index == len(self.subgoal_generator.subgoals) - 1:
                    reward = self._reward()
                    done = True
            else:   
                if self.carrying and self.carrying == self.obj:
                    reward = self._reward()
                    done = True
        
        more_info = self.subgoal_generator.check_subgoal_completion_after()
        info.update(more_info)

        if info['subgoal_completed'] and self.pretrain:
            done = True

        return obs, reward, done, info
    
    def get_current_subgoal(self):
        return self.subgoal_generator.get_current_subgoal()
    
    def get_current_subgoal_id(self):
        return self.subgoal_generator.get_current_subgoal_id()
    
    def get_current_subgoal_pos(self):
        return self.subgoal_generator.get_current_subgoal_pos()

class KeyCorridorS3R1(KeyCorridor):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            room_size=3,
            num_rows=1,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance,
        )

class KeyCorridorS3R2(KeyCorridor):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            room_size=3,
            num_rows=2,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance,
        )

class KeyCorridorS3R3(KeyCorridor):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            room_size=3,
            num_rows=3,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance,
        )

class KeyCorridorS4R3(KeyCorridor):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance,
        )

class KeyCorridorS5R3(KeyCorridor):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            room_size=5,
            num_rows=3,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance,
        )

class KeyCorridorS6R3(KeyCorridor):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            room_size=6,
            num_rows=3,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance,
        )

register(
    id='MiniGrid-KeyCorridorS3R1-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS3R1'
)

register(
    id='MiniGrid-KeyCorridorS3R2-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS3R2'
)

register(
    id='MiniGrid-KeyCorridorS3R3-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS3R3'
)

register(
    id='MiniGrid-KeyCorridorS4R3-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS4R3'
)

register(
    id='MiniGrid-KeyCorridorS5R3-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS5R3'
)

register(
    id='MiniGrid-KeyCorridorS6R3-v0',
    entry_point='gym_minigrid.envs:KeyCorridorS6R3'
)

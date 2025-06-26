from gym_minigrid.minigrid import *
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register

from .subgoal_generator import SubgoalGenerator

class ObstructedMazeEnv(RoomGrid):
    """
    A blue ball is hidden in the maze. Doors may be locked,
    doors may be obstructed by a ball and keys may be hidden in boxes.
    """

    def __init__(self,
        num_rows,
        num_cols,
        num_rooms_visited,
        seed=None
    ):
        room_size = 6
        max_steps = 4*num_rooms_visited*room_size**2

        self.episode_seed = seed

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            max_steps=max_steps,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Define all possible colors for doors
        self.door_colors = self._rand_subset(COLOR_NAMES, len(COLOR_NAMES))
        # Define the color of the ball to pick up
        self.ball_to_find_color = COLOR_NAMES[0]
        # Define the color of the balls that obstruct doors
        self.blocking_ball_color = COLOR_NAMES[1]
        # Define the color of boxes in which keys are hidden
        self.box_color = COLOR_NAMES[2]

        self.mission = "pick up the %s ball" % self.ball_to_find_color

    def step(self, action):
        self.subgoal_generator.check_subgoal_completion_before()

        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                done = True

        more_info = self.subgoal_generator.check_subgoal_completion_after()
        info.update(more_info)

        return obs, reward, done, info

    def add_door(self, i, j, door_idx=0, color=None, locked=False, key_in_box=False, blocked=False):
        """
        Add a door. If the door must be locked, it also adds the key.
        If the key must be hidden, it is put in a box. If the door must
        be obstructed, it adds a ball in front of the door.
        """

        door, door_pos = super().add_door(i, j, door_idx, color, locked=locked)
        box, box_pos = None, None
        blocking_ball, blocking_ball_pos = None, None

        if blocked:
            vec = DIR_TO_VEC[door_idx]
            blocking_ball = Ball(self.blocking_ball_color) if blocked else None
            self.grid.set(door_pos[0]-vec[0], door_pos[1]-vec[1], blocking_ball)
            blocking_ball_pos = door_pos[0]-vec[0], door_pos[1]-vec[1]
            
        if locked:
            obj = Key(door.color)
            if key_in_box:
                box = Box(self.box_color) if key_in_box else None
                box.contains = obj
                obj = box
            box, box_pos = self.place_in_room(i, j, obj)

        return door, door_pos, box, box_pos, blocking_ball, blocking_ball_pos
    
    def get_current_subgoal(self):
        return self.subgoal_generator.get_current_subgoal()
    
    def get_current_subgoal_id(self):
        return self.subgoal_generator.get_current_subgoal_id()
    
    def get_current_subgoal_pos(self):
        return self.subgoal_generator.get_current_subgoal_pos()

class ObstructedMaze_1Dlhb(ObstructedMazeEnv):
    """
    A blue ball is hidden in a 2x1 maze. A locked door separates
    rooms. Doors are obstructed by a ball and keys are hidden in boxes.
    """

    def __init__(
            self,
            key_in_box=True,
            blocked=True,
            seed=None,
            subgoal_file=None,
            subgoal_type="relative",
            subgoal_accuracy=[1],
            subgoal_mean_error=[0],
            subgoal_std_error=[0],
            n_random_subgoals=0,
            pretrain=False,
            pretrain_subgoal_distance=0,
    ):
        self.key_in_box = key_in_box
        self.blocked = blocked
        self.subgoal_file = subgoal_file
        self.subgoal_type = subgoal_type
        self.subgoal_accuracy = subgoal_accuracy
        self.subgoal_mean_error = subgoal_mean_error
        self.subgoal_std_error = subgoal_std_error

        self.n_random_subgoals = n_random_subgoals

        self.pretrain = pretrain
        self.pretrain_subgoal_distance = pretrain_subgoal_distance

        super().__init__(
            num_rows=1,
            num_cols=2,
            num_rooms_visited=2,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        door, door_pos, box, box_pos, bball, bball_pos = self.add_door(0, 0, door_idx=0, color=self.door_colors[0],
                      locked=True,
                      key_in_box=self.key_in_box,
                      blocked=self.blocked)

        self.obj, obj_pos = self.add_object(1, 0, "ball", color=self.ball_to_find_color)
        self.goal_pos = obj_pos
        self.place_agent(0, 0)

        if bball:
            subgoals = [
                ("box", list(box_pos)),
                ("ball", list(bball_pos)),
                ("door", list(door_pos)),
                ("ball", list(obj_pos)),
            ]
        else:
            subgoals = [
                ("box", list(box_pos)),
                ("door", list(door_pos)),
                ("ball", list(obj_pos)),
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
                                                  force_doors=False,
                                                  include_walls=False,
                                                  include_locked_room=False)


class ObstructedMaze_1Dl(ObstructedMaze_1Dlhb):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type="relative", subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            key_in_box=False,
            blocked=False,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance
        )

class ObstructedMaze_1Dlh(ObstructedMaze_1Dlhb):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type="relative", subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            key_in_box=True,
            blocked=False,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance
        )
        
class ObstructedMaze_Full(ObstructedMazeEnv):
    """
    A blue ball is hidden in one of the 4 corners of a 3x3 maze. Doors
    are locked, doors are obstructed by a ball and keys are hidden in
    boxes.
    """

    def __init__(self, agent_room=(1, 1), key_in_box=True, blocked=True,
                 num_quarters=4, num_rooms_visited=25, seed=None):
        self.agent_room = agent_room
        self.key_in_box = key_in_box
        self.blocked = blocked
        self.num_quarters = num_quarters

        super().__init__(
            num_rows=3,
            num_cols=3,
            num_rooms_visited=num_rooms_visited,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        middle_room = (1, 1)
        # Define positions of "side rooms" i.e. rooms that are neither
        # corners nor the center.
        side_rooms = [(2, 1), (1, 2), (0, 1), (1, 0)][:self.num_quarters]
        for i in range(len(side_rooms)):
            side_room = side_rooms[i]

            # Add a door between the center room and the side room
            self.add_door(*middle_room, door_idx=i, color=self.door_colors[i], locked=False)

            for k in [-1, 1]:
                # Add a door to each side of the side room
                self.add_door(*side_room, locked=True,
                              door_idx=(i+k)%4,
                              color=self.door_colors[(i+k)%len(self.door_colors)],
                              key_in_box=self.key_in_box,
                              blocked=self.blocked)

        corners = [(2, 0), (2, 2), (0, 2), (0, 0)][:self.num_quarters]
        ball_room = self._rand_elem(corners)

        self.obj, _ = self.add_object(*ball_room, "ball", color=self.ball_to_find_color)
        self.place_agent(*self.agent_room)

class ObstructedMaze_2Dl(ObstructedMaze_Full):
    def __init__(self, seed=None):
        super().__init__((2, 1), False, False, 1, 4, seed)

class ObstructedMaze_2Dlh(ObstructedMaze_Full):
    def __init__(self, seed=None):
        super().__init__((2, 1), True, False, 1, 4, seed)

class ObstructedMaze_2Dlhb(ObstructedMaze_Full):
    def __init__(self, seed=None):
        super().__init__((2, 1), True, True, 1, 4, seed)

class ObstructedMaze_1Q(ObstructedMaze_Full):
    def __init__(self, seed=None):
        super().__init__((1, 1), True, True, 1, 5, seed)

class ObstructedMaze_2Q(ObstructedMaze_Full):
    def __init__(self, seed=None):
        super().__init__((1, 1), True, True, 2, 11, seed)

register(
    id="MiniGrid-ObstructedMaze-1Dl-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_1Dl"
)

register(
    id="MiniGrid-ObstructedMaze-1Dlh-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_1Dlh"
)

register(
    id="MiniGrid-ObstructedMaze-1Dlhb-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_1Dlhb"
)

register(
    id="MiniGrid-ObstructedMaze-2Dl-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_2Dl"
)

register(
    id="MiniGrid-ObstructedMaze-2Dlh-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_2Dlh"
)

register(
    id="MiniGrid-ObstructedMaze-2Dlhb-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_2Dlhb"
)

register(
    id="MiniGrid-ObstructedMaze-1Q-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_1Q"
)

register(
    id="MiniGrid-ObstructedMaze-2Q-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_2Q"
)

register(
    id="MiniGrid-ObstructedMaze-Full-v0",
    entry_point="gym_minigrid.envs:ObstructedMaze_Full"
)
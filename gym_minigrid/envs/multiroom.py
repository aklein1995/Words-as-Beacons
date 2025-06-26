from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from .subgoal_generator import SubgoalGenerator

class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos

class MultiRoomEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        subgoal_file=None,
        subgoal_type='relative',
        subgoal_accuracy=[1],
        subgoal_mean_error=[0],
        subgoal_std_error=[0],
        n_random_subgoals=0,
        pretrain=False,
        pretrain_subgoal_distance=0,
        seed=None
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.subgoal_file = subgoal_file
        self.subgoal_type = subgoal_type
        self.subgoal_accuracy = subgoal_accuracy
        self.n_random_subgoals = n_random_subgoals

        self.subgoal_mean_error = subgoal_mean_error
        self.subgoal_std_error = subgoal_std_error

        self.pretrain = pretrain
        self.pretrain_subgoal_distance = pretrain_subgoal_distance

        self.rooms = []
        self.episode_seed = seed

        super(MultiRoomEnv, self).__init__(
            grid_size=25,
            max_steps=self.maxNumRooms * 20,
            seed=seed
        )

    def _dijkstra(self):
        start_point = self.agent_pos
        end_point = self.goal_pos

        num_rows, num_cols = self.grid.height, self.grid.height

        distances = [[float('inf')] * num_cols for _ in range(num_rows)]
        distances[start_point[0]][start_point[1]] = 0
        

        visited = [[False] * num_cols for _ in range(num_rows)]
        previous = {}

        def neighbors(r, c):
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < num_rows and 0 <= nc < num_cols:
                    if self.grid.get(nr, nc) and self.grid.get(nr, nc).type == "wall":
                        continue
                    yield nr, nc

        while True:
            min_distance = float('inf')
            min_vertex = None
            
            for r in range(num_rows):
                for c in range(num_cols):
                    if not visited[r][c] and distances[r][c] < min_distance:
                        min_distance = distances[r][c]
                        min_vertex = (r, c)
            
            if min_vertex is None:
                break
            
            r, c = min_vertex
            visited[r][c] = True
            
            for nr, nc in neighbors(r, c):
                if not visited[nr][nc]:
                    alt = distances[r][c] + 1
                    if alt < distances[nr][nc]:
                        distances[nr][nc] = alt
                        previous[(nr, nc)] = (r, c)

        path = []
        current = end_point
        while not np.all(current == start_point):
            path.append(current)
            current = previous[tuple(current)]
        path.append(start_point)
        path.reverse()

        return path

    def _path_to_subgoals(self, path):
        subgoals = []
        for point in path[1:-1]:
            cell = self.grid.get(*point)
            if not cell: continue

            if "door" in self.grid.get(*point).type:
                subgoals.append(("door", list(point)))
            
        subgoals.append(("goal", list(path[-1])))
        return subgoals

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        subgoals = self._path_to_subgoals(self._dijkstra())
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
                                                  include_locked_room=True)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

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
    
    def get_current_subgoal(self):
        return self.subgoal_generator.get_current_subgoal()
    
    def get_current_subgoal_id(self):
        return self.subgoal_generator.get_current_subgoal_id()
    
    def get_current_subgoal_pos(self):
        return self.subgoal_generator.get_current_subgoal_pos()
    
class MultiRoomEnvN2S4(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=4,
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

class MultiRoomEnvN4S5(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=4,
            maxNumRooms=4,
            maxRoomSize=5,
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

class MultiRoomEnvN6(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6,
            seed=seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals
        ),
        pretrain=pretrain,
        pretrain_subgoal_distance=pretrain_subgoal_distance,

class MultiRoomEnvN3S8(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=3,
            maxNumRooms=3,
            maxRoomSize=8,
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

class MultiRoomEnvN7S8(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=7,
            maxNumRooms=7,
            maxRoomSize=8,
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
class MultiRoomEnvN7S4(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=7,
            maxNumRooms=7,
            maxRoomSize=4,
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
class MultiRoomEnvN7S5(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=7,
            maxNumRooms=7,
            maxRoomSize=5,
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
class MultiRoomEnvN7S6(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=7,
            maxNumRooms=7,
            maxRoomSize=6,
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

class MultiRoomEnvN10S4(MultiRoomEnv):
    def __init__(self, seed=None, subgoal_file=None, subgoal_type='relative', subgoal_accuracy=[1], subgoal_mean_error=[0], subgoal_std_error=[0], n_random_subgoals=0, pretrain=False, pretrain_subgoal_distance=0):
        super().__init__(
            minNumRooms=10,
            maxNumRooms=10,
            maxRoomSize=4,
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
    id='MiniGrid-MultiRoom-N2-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN2S4'
)

register(
    id='MiniGrid-MultiRoom-N4-S5-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN4S5'
)
register(
    id='MiniGrid-MultiRoom-N6-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN6'
)

# size 8
register(
    id='MiniGrid-MultiRoom-N3-S8-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN3S8'
)
register(
    id='MiniGrid-MultiRoom-N7-S8-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN7S8'
)

# 7 rooms with different size
register(
    id='MiniGrid-MultiRoom-N7-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN7S4'
)
register(
    id='MiniGrid-MultiRoom-N7-S5-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN7S5'
)
register(
    id='MiniGrid-MultiRoom-N7-S6-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN7S6'
)

register(
    id='MiniGrid-MultiRoom-N10-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnvN10S4'
)

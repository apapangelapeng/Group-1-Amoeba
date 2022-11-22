import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState
import math
import time

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger, metabolism: float, goal_size: int,
                 precomp_dir: str) -> None:
        """Initialise the player with the basic amoeba information

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                metabolism (float): the percentage of amoeba cells, that can move
                goal_size (int): the size the amoeba must reach
                precomp_dir (str): Directory path to store/load pre-computation
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.metabolism = metabolism
        self.goal_size = goal_size
        self.current_size = goal_size / 4

    def move(self, last_percept, current_percept, info) -> (list, list, int):
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement

            Args:
                last_percept (AmoebaState): contains state information after the previous move
                current_percept(AmoebaState): contains current state information
                info (int): byte (ranging from 0 to 256) to convey information from previous turn
            Returns:
                Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]: This function returns three variables:
                    1. A list of cells on the periphery that the amoeba retracts
                    2. A list of positions the retracted cells have moved to
                    3. A byte of information (values range from 0 to 255) that the amoeba can use
        """



        # self.current_size = current_percept.current_size
        # mini = min(5, len(current_percept.periphery) // 2)
        # for i, j in current_percept.bacteria:
        #     current_percept.amoeba_map[i][j] = 1
        #
        # retract = [tuple(i) for i in self.rng.choice(current_percept.periphery, replace=False, size=mini)]
        # movable = self.find_movable_cells(retract, current_percept.periphery, current_percept.amoeba_map,
        #                                   current_percept.bacteria, mini)
        #
        # info = 0

        sz = current_percept.current_size
        max_movable = min(sz//2, int(math.ceil(sz * self.metabolism)))
        retract, movable = [], []

        if self.is_square(current_percept):
            for row in current_percept.amoeba_map:
                print(row)
            print('##########################')
            min_x, max_x, min_y, max_y = self.bounds(current_percept)
            first_row = []
            second_row = []
            third_row = []
            for x in range(min_x, max_x+1):
                first_row.append((min_y, x))
                # second_row.append((min_y + 1, x))
                # third_row.append((min_y + 2, x))

            target_positions = [(min_y+1, min_x-1), (min_y+2, min_x-1), (min_y+1, max_x+1), (min_y+2, max_x+1)]

            to_retract = first_row[1::2]
            for i in range(min([len(to_retract), max_movable, 4])):
                retract.append(to_retract.pop())
                movable.append(target_positions.pop())

            for y, x in current_percept.periphery:
                if target_positions and y > min_y + 2:
                    retract.append((y, x))
                    movable.append(target_positions.pop())
        else:
            time.sleep(60)
            print('**************')

        return retract, movable, info

    def bounds(self, current_percept):
        min_x, max_x, min_y, max_y = 100, -1, 100, -1
        # for i in range(100):
        #     for j in range(100):
        #         if current_percept.amoeba_map[i][j] == 1:
        #             if i < min_y:
        #                 min_y = i
        #             elif i > max_y:
        #                 max_y = i
        #             if j < min_x:
        #                 min_x = j
        #             elif j > max_x:
        #                 max_x = j
        for y, x in current_percept.periphery:
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x

        return min_x, max_x, min_y, max_y

    def is_square(self, current_percept):
        min_x, max_x, min_y, max_y = self.bounds(current_percept)
        print(min_x, max_x, min_y, max_y)
        len_x = max_x - min_x + 1
        len_y = max_y - min_y + 1
        if len_x == len_y and len_x * len_y == current_percept.current_size:
            return True
        return False

    def find_movable_cells(self, retract, periphery, amoeba_map, bacteria, mini):
        movable = []
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, amoeba_map, bacteria)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        movable += retract

        return movable[:mini]

    def find_movable_neighbor(self, x, y, amoeba_map, bacteria):
        out = []
        if (x, y) not in bacteria:
            if amoeba_map[x][(y - 1) % 100] == 0:
                out.append((x, (y - 1) % 100))
            if amoeba_map[x][(y + 1) % 100] == 0:
                out.append((x, (y + 1) % 100))
            if amoeba_map[(x - 1) % 100][y] == 0:
                out.append(((x - 1) % 100, y))
            if amoeba_map[(x + 1) % 100][y] == 0:
                out.append(((x + 1) % 100, y))

        return out

import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState


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

        self.turn = 0

    def get_left_row(self, periphery):
        # Credit: G8
        # Gets left row of amoeba to be retracted
        # For each y coord, we want the one with the lowest x coord

        top_row_vals = {} # key: y, value: x
        for (x, y) in periphery:
            if y in top_row_vals:
                if x < top_row_vals[y]:
                    top_row_vals[y] = x
            else:
                top_row_vals[y] = x

        top_row = []
        for key, val in top_row_vals.items():
            top_row.append((val, key)) # switched

        return top_row

    def get_right_row(self, periphery):
        # Credit: G8
        # Gets right row of amoeba to be retracted
        # For each y coord, we want the one with the highest x coord

        top_row_vals = {} # key: y, value: x
        for (x, y) in periphery:
            if y in top_row_vals:
                if x > top_row_vals[y]:
                    top_row_vals[y] = x
            else:
                top_row_vals[y] = x

        top_row = []
        for key, val in top_row_vals.items():
            top_row.append((val, key)) # switched

        return top_row

    def create_formation(self, last_percept, current_percept, info) -> (list, list, int):
        retract = self.get_left_row(current_percept.periphery)
        sorted_retract = sorted(retract, key=lambda x: x[1])
        mini = len(sorted_retract)
        movable = self.find_movable_cells(sorted_retract, current_percept.periphery, current_percept.amoeba_map,
                                          current_percept.bacteria, mini)
        sorted_movable_cells = sorted(movable, key=lambda x: x[1])
        return sorted_retract, sorted_movable_cells, info

        '''retract_left = self.get_left_row(current_percept.periphery) # list of tuples
        retract_right = self.get_right_row(current_percept.periphery)

        sorted_retract_left = sorted(retract_left, key=lambda x: x[1])
        sorted_retract_right = sorted(retract_right, key=lambda x: x[1])

        movable = self.find_movable_cells(sorted_retract_left + sorted_retract_right, current_percept.periphery,
                                               current_percept.amoeba_map,
                                               current_percept.bacteria, len(sorted_retract_left + sorted_retract_right))
        sorted_movable_cells = sorted(movable, key=lambda x: x[1])
        return sorted_retract_left + sorted_retract_right, sorted_movable_cells, info'''

        '''movable_left = self.find_movable_cells(sorted_retract_left, current_percept.periphery, current_percept.amoeba_map,
                                          current_percept.bacteria, len(sorted_retract_left))
        movable_right = self.find_movable_cells(sorted_retract_right, current_percept.periphery,
                                               current_percept.amoeba_map,
                                               current_percept.bacteria, len(sorted_retract_right))

        sorted_movable_cells = movable_left + movable_right #sorted(movable, key=lambda x: x[1])'''

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

        self.turn += 1
        self.current_size = current_percept.current_size
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = 1

        if self.turn < 40:
            return self.create_formation(last_percept, current_percept, info)
        else:
            # move amoeba forward
            mini = min(5, len(current_percept.periphery) // 2)
            for i, j in current_percept.bacteria:
                current_percept.amoeba_map[i][j] = 1

            retract = set()
            while len(retract) < 5:
                i = self.rng.choice(current_percept.periphery, replace=False)
                x = i[0]
                y = i[1]
                if x < 50 and y > 50:
                    retract.add(tuple(i))
            retract = list(retract)

            movable = self.find_movable_cells(retract, current_percept.periphery, current_percept.amoeba_map,
                                              current_percept.bacteria, mini)
            info = 0
            return retract, movable, info

    def find_movable_cells(self, retract, periphery, amoeba_map, bacteria, mini):
        # sort periphery by y coord
        '''sorted_periphery = sorted(periphery, key=lambda x: x[1])
        lowest_y = sorted_periphery[0][1]
        highest_y = sorted_periphery[-1][1]
        print(lowest_y, highest_y)'''

        movable = []
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, amoeba_map, bacteria)
            for x, y in nbr:
                if (x, y) not in movable and y < 52 and (x > 45 and x < 55):
                    movable.append((x, y))

        movable += retract

        return movable[:mini]

    def find_movable_neighbor(self, x, y, amoeba_map, bacteria):
        # a cell is on the periphery if it borders (orthogonally) a cell that is not occupied by the amoeba
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

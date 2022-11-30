import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy.typing as npt
import constants
import math


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

        # Class accessible percept variables, written at the start of each turn
        self.current_size: int = None
        self.amoeba_map: npt.NDArray = None
        self.bacteria_cells: List[Tuple[int, int]] = None
        self.retractable_cells: List[Tuple[int, int]] = None
        self.extendable_cells: List[Tuple[int, int]] = None
        self.num_available_moves: int = None

    
    def store_current_percept(self, current_percept: AmoebaState) -> None:
        self.current_size = current_percept.current_size
        self.amoeba_map = current_percept.amoeba_map
        self.retractable_cells = current_percept.periphery
        self.bacteria_cells = current_percept.bacteria
        self.extendable_cells = current_percept.movable_cells
        self.num_available_moves = int(np.ceil(self.metabolism * current_percept.current_size))


    def move(self, last_percept, current_percept, info) -> (List, List, int):
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
        self.store_current_percept(current_percept)
        
        # self.current_size = current_percept.current_size
        # mini = min(5, len(current_percept.periphery) // 2)
        # for i, j in current_percept.bacteria:
        #     current_percept.amoeba_map[i][j] = 1

        # Saving amoeba map to txt file
        # if self.turn == 1:
        #     x = np.array(current_percept.amoeba_map)
        #     x.astype(int)
        #     np.savetxt('test.txt', x, fmt='%d')


        info = 0

        retracts, extends = self.get_top_moves()

        print("Turn #{}".format(self.turn))
        # print("Retract: ", retract)
        print("Retract: ", retracts)
        # print("Move to: ", movable)
        print("Extends: ", extends)


        # return retract, movable, info
        return retracts, extends, info

    
    # TESTING
    def check_move(self, retracts: List[Tuple[int, int]], extends: List[Tuple[int, int]]) -> bool:
        if not set(retracts).issubset(set(self.retractable_cells)):
            return False

        movable = retracts[:]
        new_periphery = list(set(self.retractable_cells).difference(set(retracts)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j, self.amoeba_map, self.bacteria_cells)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        if not set(extends).issubset(set(movable)):
            return False

        amoeba = np.copy(self.amoeba_map)
        amoeba[amoeba < 0] = 0
        amoeba[amoeba > 0] = 1

        for i, j in retracts:
            amoeba[i][j] = 0

        for i, j in extends:
            amoeba[i][j] = 1

        tmp = np.where(amoeba == 1)
        result = list(zip(tmp[0], tmp[1]))
        check = np.zeros((constants.map_dim, constants.map_dim), dtype=int)

        stack = result[0:1]
        while len(stack):
            a, b = stack.pop()
            check[a][b] = 1

            if (a, (b - 1) % constants.map_dim) in result and check[a][(b - 1) % constants.map_dim] == 0:
                stack.append((a, (b - 1) % constants.map_dim))
            if (a, (b + 1) % constants.map_dim) in result and check[a][(b + 1) % constants.map_dim] == 0:
                stack.append((a, (b + 1) % constants.map_dim))
            if ((a - 1) % constants.map_dim, b) in result and check[(a - 1) % constants.map_dim][b] == 0:
                stack.append(((a - 1) % constants.map_dim, b))
            if ((a + 1) % constants.map_dim, b) in result and check[(a + 1) % constants.map_dim][b] == 0:
                stack.append(((a + 1) % constants.map_dim, b))

        return (amoeba == check).all()
    
    
    def get_top_moves(self):
        potential_retracts = sorted( self.retractable_cells, key=lambda x: x[1] )
        potential_extends = sorted( self.extendable_cells, key=lambda x: x[1], reverse=True )
        
        # Loop through potential extends, searching for a matching retract
        retracts = []
        extends = []
        for potential_extend in potential_extends:
            for potential_retract in potential_retracts:
                if self.check_move(retracts + [potential_retract], extends + [potential_extend]):
                    # matching retract found, add the extend and retract to our lists
                    retracts.append(potential_retract)
                    potential_retracts.remove(potential_retract)
                    extends.append(potential_extend)
                    potential_extends.remove(potential_extend)

                    if len(retracts) == self.num_available_moves:
                        return retracts[:self.num_available_moves], extends[:self.num_available_moves]

                    break
                
        # show_amoeba_map(self.amoeba_map, retracts, extends)
        return retracts[:self.num_available_moves], extends[:self.num_available_moves]
    


    


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
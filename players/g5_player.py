import os
import pickle
import random

import numpy as np
import numpy.typing as npt
from typing import Tuple, List
import logging
from amoeba_state import AmoebaState
import math
import time
import matplotlib.pyplot as plt
from enum import Enum
import sys
import random as rnd

# CONSTS

MAP_DIM = 100
MAX_BASE_LEN = min(MAP_DIM, 100)
TOOTH_SPACING = 0
SHIFTING_FREQ = 10
# MAX_BASE_LEN = min(MAP_DIM, 100)


# ********* HELPER FUNCTIONS ********* #


# the following 3 methods are borrowed from G2
def map_to_coords(amoeba_map: npt.NDArray) -> list[Tuple[int, int]]:
    return list(map(tuple, np.transpose(amoeba_map.nonzero()).tolist()))


def coords_to_map(coords: list[tuple[int, int]], size=MAP_DIM) -> npt.NDArray:
    amoeba_map = np.zeros((size, size), dtype=np.int8)
    for x, y in coords:
        amoeba_map[x, y] = 1
    return amoeba_map


def show_amoeba_map(amoeba_map: npt.NDArray, retracts=[], extends=[]) -> None:
    retracts_map = coords_to_map(retracts)
    extends_map = coords_to_map(extends)

    map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
    for x in range(MAP_DIM):
        for y in range(MAP_DIM):
            # transpose map for visualization as we add cells
            if retracts_map[x, y] == 1:
                map[y, x] = -1
            elif extends_map[x, y] == 1:
                map[y, x] = 2
            elif amoeba_map[x, y] == 1:
                map[y, x] = 1

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.pcolormesh(map, edgecolors='k', linewidth=1)
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.savefig(f"debug/{turn}.png")
    plt.show()


# the following 2 methods are borrowed from myself from project 3
def tree_index(factors, max_factors):
    index = 0
    for i in range(len(factors)):
        n_children = 1
        for factor in max_factors[i + 1:]:
            n_children *= factor
        index += factors[i] * n_children
    return index


def tree_factors(index, max_factors):
    factors = []
    for i in range(len(max_factors)):
        n_children = 1
        for factor in max_factors[i + 1:]:
            n_children *= factor
        factor = index // n_children
        factors.append(factor)
        index %= n_children

    return factors


# ********* BYTE INFO ******** #

class MaxVals(Enum):
    #is_rake = 2
    x_val = 100
    tooth_shift = 2


class Memory:
    def __init__(self, byte=None, vals=None):
        if byte is not None:
            vals = get_byte_info(byte)
            self.x_val = vals[0]
            self.tooth_shift = vals[1]
        elif vals is not None:
            self.x_val = vals[0]
            self.tooth_shift = vals[1]
        else:
            self.x_val = 50
            self.tooth_shift = 0

    def get_byte(self):
        return set_byte_info([self.x_val, self.tooth_shift])

    def get_vals(self):
        return [self.x_val, self.tooth_shift]


def get_byte_info(byte: int):
    max_vals = [field.value for field in MaxVals]
    factors = tree_factors(byte, max_vals)

    return factors


def set_byte_info(values):
    max_vals = [field.value for field in MaxVals]
    byte = tree_index(values, max_vals)

    return byte


# ********* MAIN CODE ********* #

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

        self.current_size: int = None
        self.amoeba_map: npt.NDArray = None
        self.bacteria_cells: List[Tuple[int, int]] = None
        self.retractable_cells: List[Tuple[int, int]] = None
        self.extendable_cells: List[Tuple[int, int]] = None
        self.num_available_moves: int = None

    @staticmethod
    def generate_tooth_formation(amoeba_size: int) -> npt.NDArray:
        formation = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
        center_x = MAP_DIM // 2
        center_y = MAP_DIM // 2
        spacing = TOOTH_SPACING + 1


        # find the number of complete 5-cell modules
        complete_modules = amoeba_size // 5
        # find the number of unfinished sections
        additional_sections = amoeba_size % 5 // 2
        # find whether there is an odd cell
        reserve = amoeba_size % 5 % 2

        base_len = min(complete_modules * 2 + additional_sections, MAX_BASE_LEN)
        # teeth_len = min(complete_modules, base_len//2)
        # teeth_len = min(complete_modules, base_len // spacing)
        teeth_len = base_len//spacing
        reserve = max(reserve, amoeba_size-(base_len*2 + teeth_len))
        start_y = center_y - base_len // 2

        # add the 2-cell-wide base
        for y in range(start_y, start_y + base_len):
            formation[center_x, y] = 1
            formation[center_x - 1, y] = 1

        # add the teeth
        start_modules = start_y  # +(additional_sections+1)//2
        for y in range(start_modules, start_y + teeth_len * spacing):
            # if (y - start_modules) % spacing == 0:
            if y % spacing == 0:
                formation[center_x + 1, y] = 1

        # # add the "reserve" cell at the back
        for i in range(reserve):
            row = i//base_len
            col = i % base_len
            formation[center_x-2-row, start_y+col] = 1
        # if reserve:
        #     formation[center_x - 2, start_y] = 1

        # show_amoeba_map(formation)
        return formation

    # copied from G2
    def get_morph_moves(self, desired_amoeba: npt.NDArray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """ Function which takes a starting amoeba state and a desired amoeba state and generates a set of retracts and extends
            to morph the amoeba shape towards the desired shape.
        """

        current_points = map_to_coords(self.amoeba_map)
        desired_points = map_to_coords(desired_amoeba)

        potential_retracts = [p for p in list(set(current_points).difference(set(desired_points))) if
                              p in self.retractable_cells]
        potential_extends = [p for p in list(set(desired_points).difference(set(current_points))) if
                             p in self.extendable_cells]

        potential_retracts.sort(key=lambda pos: pos[0])
        potential_extends.sort(key=lambda pos: pos[0])

        #print("Potential Retracts", potential_retracts)
        #print("Potential Extends", potential_extends)

        # Change this since it breaks the situation with smaller metabolisms
        # if len(potential_retracts) > self.num_available_moves:
        #     return [], []

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
                    break

        # show_amoeba_map(self.amoeba_map, retracts, extends)
        return retracts[:self.num_available_moves], extends[:self.num_available_moves]

    # adapted from amoeba game code
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
        check = np.zeros((MAP_DIM, MAP_DIM), dtype=int)

        stack = result[0:1]
        while len(stack):
            a, b = stack.pop()
            check[a][b] = 1

            if (a, (b - 1) % MAP_DIM) in result and check[a][(b - 1) % MAP_DIM] == 0:
                stack.append((a, (b - 1) % MAP_DIM))
            if (a, (b + 1) % MAP_DIM) in result and check[a][(b + 1) % MAP_DIM] == 0:
                stack.append((a, (b + 1) % MAP_DIM))
            if ((a - 1) % MAP_DIM, b) in result and check[(a - 1) % MAP_DIM][b] == 0:
                stack.append(((a - 1) % MAP_DIM, b))
            if ((a + 1) % MAP_DIM, b) in result and check[(a + 1) % MAP_DIM][b] == 0:
                stack.append(((a + 1) % MAP_DIM, b))

        return (amoeba == check).all()

    # copied from G2
    def store_current_percept(self, current_percept: AmoebaState) -> None:
        self.current_size = current_percept.current_size
        self.amoeba_map = current_percept.amoeba_map
        self.retractable_cells = current_percept.periphery
        self.bacteria_cells = current_percept.bacteria
        self.extendable_cells = current_percept.movable_cells
        self.num_available_moves = int(np.ceil(self.metabolism * current_percept.current_size))

    def move(self, last_percept: AmoebaState, current_percept: AmoebaState, info: int) -> Tuple[
        List[Tuple[int, int]], List[Tuple[int, int]], int]:
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

        self.store_current_percept(current_percept)

        retracts = []
        moves = []
        mem = Memory(byte=info)

        # memory_fields = read_memory(info)
        # if not memory_fields[MemoryFields.Initialized]:

        #if not mem.is_rake:
            # retracts, moves = self.get_morph_moves(self.generate_tooth_formation(self.current_size))
            # print(len(moves))
            # if len(moves) == 0:
                # info = change_memory_field(info, MemoryFields.Initialized, True)
                #mem.is_rake = True
                #info = mem.get_byte()
                # memory_fields = read_memory(info)
        #if mem.is_rake:
            # print("hi")
        # TODO: implement this (moves when the amoeba is a rake)
        # comb_col = np.argwhere(self.amoeba_map == 1)
        
        # comb_start = comb_col[0][0]
        # comb_end = comb_col[-1][0]
        
        # cells_to_move = abs(comb_end-comb_start)
        
        if self.is_square(current_percept):
            mem.x_val = 50
            mem.tooth_shift = 0

        while len(retracts) == 0 and len(moves) == 0:
            offset_x = mem.x_val - MAP_DIM//2
            offset_y = 0 if (mem.x_val % SHIFTING_FREQ*2) < SHIFTING_FREQ else -1
            # if random.random() < 0.5:
            #     mem.tooth_shift ^= 1
            # #
            # offset_y = -mem.tooth_shift
            
            target_formation = self.generate_tooth_formation(self.current_size)
            target_formation = np.roll(target_formation, offset_x, 0)
            # target_formation = self.shift_col(target_formation, mem.x_val+1, offset_y)
            target_formation = np.roll(target_formation, offset_y, 1)
            retracts, moves = self.get_morph_moves(target_formation)

            if len(retracts) == 0 and len(moves) == 0:
                mem.x_val = (mem.x_val + 1) % 100
                print('--------------------')
            # elif mem.x_val > 55 and rnd.choices([True, False], weights=(0.01, 0.99)):
            #     mem.x_val = (mem.x_val + 1) % 100
            
        # print(cells_to_move)
        # if (self.num_available_moves // cells_to_move > 0):
        #     # amount of cols the comb should progress
        #     next_tooth = np.roll(self.amoeba_map, 1, 0)
        #     self.down_1(next_tooth, )
        #     mem.is_rake = False
        #     info = mem.get_byte()
        # else:
        #     next_tooth = self.amoeba_map

        
        # retracts, moves = self.get_morph_moves(next_tooth)
        #print(retracts,  moves)
            #time.sleep(60)

        # if memory_fields[MemoryFields.Initialized]:
        # if mem.is_rake:
        #     curr_backbone_col = min(x for x, _ in map_to_coords(self.amoeba_map))
        #     vertical_shift = curr_backbone_col % 2
        #     offset = (curr_backbone_col + 1) - (MAP_DIM // 2)
        #     next_tooth = np.roll(self.generate_tooth_formation(self.current_size), offset + 1, 0)
        #     # Shift up/down by 1 every other column
        #     next_tooth = np.roll(next_tooth, vertical_shift, 1)
        #     retracts, moves = self.get_morph_moves(next_tooth)
        #     print(retracts, moves)
        info = mem.get_byte()
        return retracts, moves, info

    def shift_col(self, arr, col, shift):
        arr2 = np.copy(arr)
        arr2[:,col] = np.roll(arr2[:,col], shift)
        return arr2
    # def move(self, last_percept, current_percept, info) -> (list, list, int):
    #     """Function which retrieves the current state of the amoeba map and returns an amoeba movement
    #
    #         Args:
    #             last_percept (AmoebaState): contains state information after the previous move
    #             current_percept(AmoebaState): contains current state information
    #             info (int): byte (ranging from 0 to 256) to convey information from previous turn
    #         Returns:
    #             Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]: This function returns three variables:
    #                 1. A list of cells on the periphery that the amoeba retracts
    #                 2. A list of positions the retracted cells have moved to
    #                 3. A byte of information (values range from 0 to 255) that the amoeba can use
    #     """
    #
    #     # self.current_size = current_percept.current_size
    #     # mini = min(5, len(current_percept.periphery) // 2)
    #     # for i, j in current_percept.bacteria:
    #     #     current_percept.amoeba_map[i][j] = 1
    #     #
    #     # retract = [tuple(i) for i in self.rng.choice(current_percept.periphery, replace=False, size=mini)]
    #     # movable = self.find_movable_cells(retract, current_percept.periphery, current_percept.amoeba_map,
    #     #                                   current_percept.bacteria, mini)
    #     #
    #     # info = 0
    #
    #     sz = current_percept.current_size
    #     max_movable = min(sz//2, int(math.ceil(sz * self.metabolism)))
    #     retract, movable = [], []
    #
    #     if self.is_square(current_percept):
    #         for row in current_percept.amoeba_map:
    #             print(row)
    #         # print('##########################')
    #         min_x, max_x, min_y, max_y = self.bounds(current_percept)
    #         first_row = []
    #         for x in range(min_x, max_x+1):
    #             first_row.append((min_y, x))
    #
    #         target_positions = [(min_y+1, min_x-1), (min_y+2, min_x-1), (min_y+1, max_x+1), (min_y+2, max_x+1)]
    #
    #         to_retract = first_row[1::2]
    #         for i in range(min([len(to_retract), max_movable, 4])):
    #             retract.append(to_retract.pop())
    #             movable.append(target_positions.pop())
    #
    #         for y, x in current_percept.periphery:
    #             if target_positions and y > min_y + 2:
    #                 retract.append((y, x))
    #                 movable.append(target_positions.pop())
    #     else:
    #         # time.sleep(60)
    #         # print('**************')
    #
    #     return retract, movable, info

    def bounds(self, current_percept):
        min_x, max_x, min_y, max_y = 100, -1, 100, -1
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
        #print(min_x, max_x, min_y, max_y)
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


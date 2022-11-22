import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState
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
        self.current_size = current_percept.current_size
        mini = min(5, len(current_percept.periphery) // 2)
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = 1

        retract = [tuple(i) for i in self.rng.choice(current_percept.periphery, replace=False, size=mini)]
        movable = self.find_movable_cells(retract, current_percept.periphery, current_percept.amoeba_map,
                                          current_percept.bacteria, mini)

        info = 0

        retract = self.retract_rear_end(movable)
        movable = self.extend_front_end(movable)
        print("retract",retract)
        print("movable",movable)
        return retract, movable, info
    def getPivotElement(self,array, left, right):
        if right < left:
            return -1
        if right == left:
            return left
        middle = None
        if (left + right) %2 ==0:
            middle = left + right//2
        else:
            middle = left + right//2 -1

        if (middle < right) and (array[middle] > array[middle + 1]):
            return middle
        if (middle > left )and (array[middle] < array[middle - 1]):
            return middle-1
        if array[left] >= array[middle]:
            return self.getPivotElement(array, left, middle-1)
        else:
            return self.getPivotElement(array, middle + 1, right)
    def retract_rear_end(self,moveable)-> list:
        #just gonna try to move the right
        all_moveable = moveable
        all_moveable.sort()
        x_index = np.array(all_moveable)[: ,0]
        pivot = None
        
        for i in range(len(x_index)-1):
            first_num = x_index[i]
            second_num = x_index[i+1]
            if second_num - first_num >1:
                pivot = i
                break
        pivot = self.getPivotElement(all_moveable,0,len(all_moveable)-1)

        num_cell_moveabel = math.ceil(self.current_size * self.metabolism)
        before_pivot = None
        if pivot == -1:
            before_pivot = num_cell_moveabel
        else:
            before_pivot = min(num_cell_moveabel,num_cell_moveabel-pivot)

        retract = all_moveable[:before_pivot]
        cell_left =int(num_cell_moveabel -  pivot)

        retract.append(all_moveable[:cell_left])
    
        retract = all_moveable[:num_cell_moveabel]
        return retract[len(retract)//2 * -1:]
    def extend_front_end(self,moveable) -> list:
        all_moveable = moveable
        all_moveable.sort()
        x_index = np.array(all_moveable)[:, 0]
        pivot = None
        for i in range(len(x_index)-1):
            first_num = x_index[i]
            second_num = x_index[i+1]
            if second_num - first_num >1:
                pivot = i
                break
        pivot = self.getPivotElement(all_moveable,0,len(all_moveable)-1)
        num_cell_moveabel = math.ceil(self.current_size * self.metabolism)
        before_pivot = None
        if pivot == -1:
            before_pivot = num_cell_moveabel
        else:
            before_pivot = min(num_cell_moveabel,num_cell_moveabel-pivot)

        retract = all_moveable[before_pivot:]
        cell_left =int(num_cell_moveabel -  pivot)
        print("all moveable",all_moveable)
        retract+= (all_moveable[:cell_left])


        extend = []
        move_y= False
        pivot = self.getPivotElement(all_moveable,0,len(all_moveable)-1)
        if pivot == len(all_moveable)-2:
            move_y =True
        for i in retract:
            if move_y:
                x = i[0]
                y =( i[1]+1) % 100
            else:
                x = i[0]+1
                y = i[1]
            
            extend.append((x,y))
        return extend[len(extend)//2 ]
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

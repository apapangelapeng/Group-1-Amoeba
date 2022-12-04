import os
import pickle
import numpy as np
import logging
from amoeba_state import AmoebaState
import math 
from statistics import mode
import numpy.typing as npt
from typing import Tuple, List
import time

MAP_DIM = 100

def map_to_coords(amoeba_map: npt.NDArray) -> list[Tuple[int, int]]:
    return list(map(tuple, np.transpose(amoeba_map.nonzero()).tolist()))

def coords_to_map(coords: list[tuple[int, int]], size=MAP_DIM) -> npt.NDArray:
    amoeba_map = np.zeros((size, size), dtype=np.int8)
    for x, y in coords:
        amoeba_map[x, y] = 1
    return amoeba_map



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
        self.teeth_length = 1 # hyper parameter
        self.teeth_gap = 2 # hyper parameter
        self.acceptable_similarity = 0.8 # how similar the ideal format and the current shape should be before we start to move
        logger.info(f"initalizing player 1, with initalize size :{ goal_size/4},teeth_length:{self.teeth_length}" )
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

        # TODO: add teeth shift

        current_size = current_percept.current_size
        
        # print('----------')
        # print(current_percept.current_size)
        # print((current_percept.amoeba_map))
        # print(current_percept.periphery)
        # print(current_percept.bacteria)
        # print(current_percept.movable_cells)


        mini = min(5, len(current_percept.periphery) // 2)
        for i, j in current_percept.bacteria:
            current_percept.amoeba_map[i][j] = 1

        retract = [tuple(i) for i in self.rng.choice(current_percept.periphery, replace=False, size=mini)]
        """ movable = self.find_movable_cells(retract, current_percept.periphery, current_percept.amoeba_map,
                                          current_percept.bacteria, mini) 
        retrac = self.retract_rear_end(movable)
        movable = self.extend_front_end() """

        movable_location = current_percept.movable_cells
        periphery = current_percept.periphery
        infoFields = InfoMem(infobits=info)     # initially 0
        #print(infoFields.pivot)
        #print(infoFields.teeth_shifted)


        if self.is_square(current_percept):
            # print("first step")
            upper_right = self.find_upper_right(periphery, -1) 
            infoFields.pivot = int(upper_right[0]) # change the info of the upper right corner

            #print(infoFields.pivot)
            #print(infoFields.teeth_shifted)

            # print("info type ", type(info)) 

        else:
            upper_right = self.find_upper_right(periphery, infoFields.pivot)
        
            #print(upper_right)
        comb_formation = self.give_comb_formation(current_size, upper_right, self.teeth_length, self.teeth_gap)
        # if the comb is about 90% formed then we can start moving 
        #print("current_size", current_size )
        

        #print(self.percentage_covered(comb_formation, periphery), self.acceptable_similarity)
        if self.movable(comb_formation,periphery):
            print("moving!")

            infoFields.pivot -= 1
            infoFields.pivot %= 100
            upper_right = (infoFields.pivot, upper_right[1]) # move left 1
            comb_formation = self.give_comb_formation(current_size, upper_right, self.teeth_length,self.teeth_gap)
            
        # print(upper_right)
        moveable_cell_num = math.ceil(self.metabolism* current_size)
        retract, extend = self.move_formation(moveable_cell_num, periphery, movable_location, comb_formation,current_size)
        """print("comb_formation=", comb_formation)
        
        print("movable_location=", movable_location)
        print("periphery=", periphery)
        print("retract=", retract)"""
        self.logger.info("extend=")#, extend)


        info = infoFields.store_info_details(infoFields.pivot, infoFields.teeth_shifted)
        return  retract, extend, info

    def movable (self, comb_formation,periphery):
        periphery_set = {tuple(x) for x in periphery} 
        comb_formation_set = {tuple(x) for x in comb_formation} 
        over_lap = periphery_set & comb_formation_set ## what are the cells that are on point
        if over_lap == comb_formation_set:
            return True
        else:
            print("overlaplength vs comb_formation Length",len(over_lap),len(comb_formation_set))
            return False
    
    def give_comb_formation(self, cell_num: int, upper_right: (int, int), teeth_length: int, teeth_gap:int)-> list[(int, int)]:
        ## (x,y), (x+1,y)
        #print("Number of cells we have", cell_num)
        #teeth_gap += 1 
        cell_num -= 1 
        cur_point_x = upper_right[0]
        cur_point_y = upper_right[1]
        formation = []
        length_left = teeth_length
        formation.append((cur_point_x, cur_point_y))
        gap = False
        gap_left = teeth_gap
        to_right = True

        while True > 0:
            if not(cell_num > 0):
                break
            if  gap == False:
                # extending the teeth
                if length_left > 0:
                    cur_point_x -= 1
                    cur_point_x %= 100
                    formation.append((cur_point_x, cur_point_y))
                    length_left -= 1
                    cell_num -= 1
                
                else:
                    #go down one extending the length of the comb
                    gap = True
                    cur_point_x += teeth_length # restore the x-axis
                    cur_point_x %= 100
                    length_left = teeth_length
            else:
                if gap_left > 0: 
                    if  to_right == False:
                        cur_point_y -= 1
                        cur_point_x += 1 # move the coordinate back 
                        cur_point_x %= 100
                        cur_point_y %= 100
                        formation.append((cur_point_x, cur_point_y))
                        gap_left -= 1
                        cell_num -= 1
                        to_right = True
                        
                    else:
                        cur_point_x -= 1 # move the coordinate back 
                        cur_point_x %= 100
                        if (cur_point_x, cur_point_y) not in formation:  
                            # print("duplicate!")
                            formation.append((cur_point_x, cur_point_y))
                            cell_num -= 1
                        to_right = False
                         
                else:
                    to_right = True
                    gap = False
                    gap_left = teeth_gap
        formation_set = list(set(map(tuple, formation)))
        #print("size of future comb", len(formation_set))

        return formation


    def find_upper_right(self,formation:list[(int, int)], info)-> (int, int):
        # going to find the pivot first, then find the upper right corner
        xs, ys = zip(*formation)
        if info == -1:
            x_coord = max(xs)
        else:
            x_coord = info
        y_coord = max(ys)
        if (x_coord, y_coord) in formation:
            return (x_coord, y_coord)
        else:
            possible_points = []
            for y in ys:
                if (x_coord, y) in formation:
                    possible_points.append((x_coord, y))
            x_can, y_can = zip(*possible_points)
            y_coord = max(y_can)
            return (x_coord, y_coord)
    
    def move_formation(self, num_movable_cell, movable_cell:list[(int,int)], movable_location:list[(int,int)], final_formation:list[(int,int)],current_size:int):
        movable_cell_set =  {tuple(x) for x in movable_cell} 
        final_formation_set ={tuple(x) for x in final_formation} 
        movable_location_set = {tuple(x) for x in movable_location} 

        cells_not_on_spot = movable_cell_set & movable_cell_set.symmetric_difference(final_formation_set)
        # print(cells_not_on_spot)
        cells_not_on_spot = list(cells_not_on_spot)
        cells_not_on_spot.sort()
        # print("cells_not_on_spot",cells_not_on_spot)
        # print("wanting_to_move",final_formation_set & movable_cell_set.symmetric_difference(final_formation_set) )
        destination = (final_formation_set & movable_cell_set.symmetric_difference(final_formation_set)) & movable_location_set
         
        destination = list(destination)
        destination.sort()
        retract=[]
        extend = []
        for i in range(min(num_movable_cell,len(cells_not_on_spot),len(destination))):
            retract.append(cells_not_on_spot[i])
            extend.append(destination[i])
        formation = np.zeros((MAP_DIM,MAP_DIM),dtype=int)
        """print("cells_not_on_spot",cells_not_on_spot)
        print("destination",destination)
        if len(retract) <= math.floor(current_size*0.001):
            print("moving")
            self.movable = True"""
        return retract, extend

    """borrowed from group 5"""
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
    


class InfoMem:
    def __init__(self, infobits=None):
        # Storing upper right value and shifting parameter. 7 bits + 1 bit
        if infobits is not None:
            if infobits == 0: # initial step
                self.pivot, self.teeth_shifted = self.get_info_details(infobits)             
            else:
                self.pivot, self.teeth_shifted= self.get_info_details(infobits) # pivot = upper_right data                
        else:
             self.pivot, self.teeth_shifted = 52, 1

    def get_info_details(self, information: int) -> (int, int):
        """Split infobits apart from infromation_int
            Intake:
                infobits (int): grouped up bits of information in int format
            Outtake:
                Tuple (pivot, teeth_shifted) split apart info
        """
        info_bits = "{0:b}".format(information).zfill(8)

        return int(info_bits[0:7], 2), int(info_bits[7:8], 2)

    def store_info_details(self, pivot: int, teeth_shifted: int) -> int:
        """ Stick infobits together and return a integer
            Intake:
                pivot (int): 7 bits for location of upper right of amoeba
                teeth_shifted (int): 1 bit to store teeth_shifting parameter
                # TODO: can implement other algorithm to create more space if required
            Output:
                int: the encoded information as an int
        """
        assert pivot >= 0 and pivot <= 100 # upper right x val is on the board
        information_int = "{:07b}{:01b}".format(pivot, teeth_shifted)

        return int(information_int, 2)


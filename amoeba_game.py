import os
import time
import signal
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from amoeba_state import AmoebaState
import constants
from utils import *
from glob import glob
from players.default_player import Player as DefaultPlayer
from players.g1_player import Player as G1_Player
from players.g2_player import Player as G2_Player
from players.g3_player import Player as G3_Player
from players.g4_player import Player as G4_Player
from players.g5_player import Player as G5_Player
from players.g6_player import Player as G6_Player
from players.g7_player import Player as G7_Player
from players.g8_player import Player as G8_Player


class AmoebaGame:
    def __init__(self, args):
        self.start_time = time.time()
        self.use_gui = not args.no_gui
        self.use_vid = not args.no_vid
        self.do_logging = not args.disable_logging
        if not self.use_gui:
            self.use_timeout = not args.disable_timeout
        else:
            self.use_timeout = False

            os.makedirs("render", exist_ok=True)

            old_files = glob("render/*.png")
            for f in old_files:
                os.remove(f)

        self.logger = logging.getLogger(__name__)
        # create file handler which logs even debug messages
        if self.do_logging:
            self.logger.setLevel(logging.DEBUG)
            self.log_dir = args.log_path
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.log_dir, 'debug.log'), mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(message)s'))
            fh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(fh)
            result_path = os.path.join(self.log_dir, "results.log")
            rfh = logging.FileHandler(result_path, mode="w")
            rfh.setLevel(logging.INFO)
            rfh.setFormatter(logging.Formatter('%(message)s'))
            rfh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(rfh)
        else:
            if args.log_path:
                self.logger.setLevel(logging.INFO)
                result_path = args.log_path
                self.log_dir = os.path.dirname(result_path)
                if self.log_dir:
                    os.makedirs(self.log_dir, exist_ok=True)
                rfh = logging.FileHandler(result_path, mode="w")
                rfh.setLevel(logging.INFO)
                rfh.setFormatter(logging.Formatter('%(message)s'))
                rfh.addFilter(MainLoggingFilter(__name__))
                self.logger.addHandler(rfh)
            else:
                self.logger.setLevel(logging.ERROR)
                self.logger.disabled = True

        if args.seed == 0:
            args.seed = None
            self.logger.info("Initialise random number generator with no seed")
        else:
            self.logger.info("Initialise random number generator with seed {}".format(args.seed))

        self.rng = np.random.default_rng(args.seed)

        self.player = None
        self.player_name = None

        self.metabolism = args.metabolism
        self.start_size = args.size
        self.amoeba_size = self.start_size ** 2
        self.goal_size = self.amoeba_size * 4
        self.goal_reached = False
        self.turns = 0
        self.max_turns = args.final
        self.game_end = self.max_turns
        self.density = args.density
        self.bacteria = []
        self.map_state = np.zeros((constants.map_dim, constants.map_dim), dtype=int)

        self.after_last_move = None
        self.player_byte = 0
        self.history = []

        self.initialize(args.size)
        self.add_player(args.player)
        self.play_game()
        self.end_time = time.time()

        print("\nTime taken: {}\n".format(self.end_time - self.start_time))

        if self.use_vid:
            if not self.use_gui:
                print("Rendering Frames...")
                self.frame_rendering_post()
                final_time = time.time()
                print("\nTime taken to render frames: {}\n".format(final_time - self.end_time))
            print("Creating Video...\n")
            os.system(
                "convert -delay 5 -loop 0 $(ls -1 render/*.png | sort -V) -quality 95 {}.mp4".format(args.vid_name))

        if self.use_gui:
            plt.show()

    def add_player(self, player_in):
        if player_in in constants.possible_players:
            if player_in.lower() == 'd':
                player_class = DefaultPlayer
                player_name = "Default Player"
            else:
                player_class = eval("G{}_Player".format(player_in))
                player_name = "Group {}".format(player_in)

            self.logger.info(
                "Adding player {} from class {}".format(player_name, player_class.__module__))
            precomp_dir = os.path.join("precomp", player_name)
            os.makedirs(precomp_dir, exist_ok=True)

            start_time = 0
            is_timeout = False
            if self.use_timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(constants.timeout)
            try:
                start_time = time.time()
                player = player_class(rng=self.rng, logger=self.get_player_logger(player_name),
                                      metabolism=self.metabolism, goal_size=self.goal_size, precomp_dir=precomp_dir)
                if self.use_timeout:
                    signal.alarm(0)  # Clear alarm
            except TimeoutException:
                is_timeout = True
                player = None
                self.logger.error(
                    "Initialization Timeout {} since {:.3f}s reached.".format(player_name, constants.timeout))

            init_time = time.time() - start_time

            if not is_timeout:
                self.logger.info("Initializing player {} took {:.3f}s".format(player_name, init_time))
            self.player = player
            self.player_name = player_name

        else:
            self.logger.error("Failed to insert player {} since invalid player name provided.".format(player_in))

    def get_player_logger(self, player_name):
        player_logger = logging.getLogger("{}.{}".format(__name__, player_name))

        if self.do_logging:
            player_logger.setLevel(logging.INFO)
            # add handler to self.logger with filtering
            player_fh = logging.FileHandler(os.path.join(self.log_dir, '{}.log'.format(player_name)), mode="w")
            player_fh.setLevel(logging.DEBUG)
            player_fh.setFormatter(logging.Formatter('%(message)s'))
            player_fh.addFilter(PlayerLoggingFilter(player_name))
            self.logger.addHandler(player_fh)
        else:
            player_logger.setLevel(logging.ERROR)
            player_logger.disabled = True

        return player_logger

    def initialize(self, sl):
        for i in range(sl):
            for j in range(sl):
                if i == 0 or i == (sl - 1) or j == 0 or j == (sl - 1):
                    self.map_state[50 - (sl // 2) + i][50 - (sl // 2) + j] = 2
                else:
                    self.map_state[50 - (sl // 2) + i][50 - (sl // 2) + j] = 1

        self.bacteria = [tuple(i) for i in self.rng.choice(self.find_indices(0), replace=False, size=math.floor(
            self.density * (constants.total_cells - self.amoeba_size)))]

        for i, j in self.bacteria:
            self.map_state[i][j] = -1

        if self.use_gui:
            self.frame_rendering()
        elif self.use_vid:
            self.history.append(self.get_state())

        periphery, eatable_bacteria, movable_cells, amoeba = self.get_periphery_info(False)
        self.after_last_move = AmoebaState(self.amoeba_size, amoeba, periphery, eatable_bacteria, movable_cells)

    def find_indices(self, value):
        result = np.where(self.map_state == value)
        return list(zip(result[0], result[1]))

    def play_game(self):
        while self.turns != self.max_turns:
            self.turns += 1
            self.play_turn()
            print("Turn {} complete".format(self.turns))
            if self.amoeba_size >= self.goal_size:
                self.goal_reached = True
                self.game_end = self.turns
                print("Goal size achieved!\n\nTurns taken: {}\nFinal size: {}\nGoal size: {}".format(self.turns,
                                                                                                     self.amoeba_size,
                                                                                                     self.goal_size))
                break

        if not self.goal_reached:
            print("Goal size not achieved...\n\nFinal size: {}\nGoal size: {}".format(self.amoeba_size, self.goal_size))

    def play_turn(self):
        self.bacteria_move()
        periphery, eatable_bacteria, movable_cells, amoeba = self.get_periphery_info(True)
        before_state = AmoebaState(self.amoeba_size, amoeba, periphery, eatable_bacteria, movable_cells)
        returned_action = self.player.move(
            last_percept=self.after_last_move,
            current_percept=before_state,
            info=self.player_byte
        )
        self.eat_bacteria(eatable_bacteria)
        if self.check_action(returned_action):
            retract, move, self.player_byte = returned_action
            if self.check_move(retract, move, periphery):
                print("Move Accepted!")
                self.logger.debug("Received move from {}".format(self.player_name))
                self.amoeba_move(retract, move)
            else:
                print("Valid move, but causes separation, hence cancelled.")
                self.logger.info("Invalid move from {} as it does not follow the rules".format(self.player_name))
        else:
            print("Invalid move")
            self.logger.info("Invalid move from {} as it doesn't follow the return format".format(self.player_name))

        self.add_bacteria()

        if self.use_gui:
            self.frame_rendering()
        elif self.use_vid:
            self.history.append(self.get_state())

        periphery, eatable_bacteria, movable_cells, amoeba = self.get_periphery_info(False)
        self.after_last_move = AmoebaState(self.amoeba_size, amoeba, periphery, eatable_bacteria, movable_cells)

    def bacteria_move(self):
        for i, (x, y) in enumerate(self.bacteria):
            avail = {'up': self.map_state[x][(y - 1) % constants.map_dim] == 0,
                     'down': self.map_state[x][(y + 1) % constants.map_dim] == 0,
                     'left': self.map_state[(x - 1) % constants.map_dim][y] == 0,
                     'right': self.map_state[(x + 1) % constants.map_dim][y] == 0}
            free_cells = [i for i in list(avail.keys()) if avail[i]]
            move = None
            if len(free_cells) == 2:
                move = self.rng.choice(free_cells, replace=False)
            elif len(free_cells) == 3:
                if 'up' in free_cells and 'down' in free_cells:
                    move = free_cells[-1]
                else:
                    move = free_cells[0]

            if move:
                self.map_state[x][y] = 0
                if move == 'up':
                    y = (y - 1) % constants.map_dim
                elif move == 'down':
                    y = (y + 1) % constants.map_dim
                elif move == 'left':
                    x = (x - 1) % constants.map_dim
                else:
                    x = (x + 1) % constants.map_dim

                self.map_state[x][y] = -1
                self.bacteria[i] = (x, y)

    def get_periphery_info(self, edit):
        periphery = self.find_indices(2)
        eatable_bacteria = []
        movable_cells = []
        rem_idx = []
        for i, j in periphery:
            nbr = self.find_movable_neighbor(i, j)
            rem = True
            for x, y in nbr:
                if (x, y) not in eatable_bacteria and (x, y) not in movable_cells:
                    if self.map_state[x][y] == -1:
                        eatable_bacteria.append((x, y))
                    else:
                        rem = False
                        movable_cells.append((x, y))
                elif self.map_state[x][y] == 0:
                    rem = False

            if rem and edit:
                self.map_state[i][j] = 1
                rem_idx.append((i, j))

        periphery = list(set(periphery).difference(set(rem_idx)))

        amoeba = np.copy(self.map_state)
        amoeba[amoeba < 0] = 0
        amoeba[amoeba > 0] = 1

        return periphery, eatable_bacteria, movable_cells, amoeba

    def find_movable_neighbor(self, x, y):
        out = []
        if self.map_state[x][(y - 1) % constants.map_dim] < 1:
            out.append((x, (y - 1) % constants.map_dim))
        if self.map_state[x][(y + 1) % constants.map_dim] < 1:
            out.append((x, (y + 1) % constants.map_dim))
        if self.map_state[(x - 1) % constants.map_dim][y] < 1:
            out.append(((x - 1) % constants.map_dim, y))
        if self.map_state[(x + 1) % constants.map_dim][y] < 1:
            out.append(((x + 1) % constants.map_dim, y))

        return out

    def find_neighbor(self, x, y, val):
        out = []
        if self.map_state[x][(y - 1) % constants.map_dim] == val:
            out.append((x, (y - 1) % constants.map_dim))
        if self.map_state[x][(y + 1) % constants.map_dim] == val:
            out.append((x, (y + 1) % constants.map_dim))
        if self.map_state[(x - 1) % constants.map_dim][y] == val:
            out.append(((x - 1) % constants.map_dim, y))
        if self.map_state[(x + 1) % constants.map_dim][y] == val:
            out.append(((x + 1) % constants.map_dim, y))

        return out

    def eat_bacteria(self, bacteria):
        for i, j in bacteria:
            self.bacteria.remove((i, j))
            self.map_state[i][j] = 2
            self.amoeba_size += 1

    def check_action(self, action):
        if not action:
            return False
        if type(action) is not tuple:
            return False
        if len(action) != 3:
            return False
        if type(action[2]) is not int:
            return False
        if action[2] < 0 or action[2] >= 256:
            return False
        if type(action[0]) is not list or type(action[1]) is not list:
            return False
        if len(action[0]) != len(set(action[0])) or len(action[1]) != len(set(action[1])):
            return False
        if len(action[0]) != len(action[1]) or len(action[0]) > math.ceil(self.metabolism * self.amoeba_size):
            return False

        return True

    def check_move(self, retract, move, periphery):
        if not set(retract).issubset(set(periphery)):
            return False

        movable = retract[:]
        new_periphery = list(set(periphery).difference(set(retract)))
        for i, j in new_periphery:
            nbr = self.find_movable_neighbor(i, j)
            for x, y in nbr:
                if (x, y) not in movable:
                    movable.append((x, y))

        if not set(move).issubset(set(movable)):
            return False

        amoeba = np.copy(self.map_state)
        amoeba[amoeba < 0] = 0
        amoeba[amoeba > 0] = 1

        for i, j in retract:
            amoeba[i][j] = 0

        for i, j in move:
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

    def amoeba_move(self, retract, move):
        for i, j in retract:
            self.map_state[i][j] = 0
            nbr = self.find_neighbor(i, j, 1)
            for x, y in nbr:
                self.map_state[x][y] = 2

        for i, j in move:
            self.map_state[i][j] = 2
            nbr = self.find_neighbor(i, j, 2)
            for x, y in nbr:
                if len(self.find_movable_neighbor(x, y)) == 0:
                    self.map_state[x][y] = 1

    def add_bacteria(self):
        new_bacteria = [tuple(i) for i in self.rng.choice(self.find_indices(0), replace=False, size=math.floor(
            self.density * (constants.total_cells - self.amoeba_size)) - len(self.bacteria))]
        self.bacteria += new_bacteria
        for i, j in new_bacteria:
            self.map_state[i][j] = -1

    def get_state(self):
        return_dict = dict()
        return_dict['amoeba_size'] = self.amoeba_size
        return_dict['bacteria'] = self.bacteria[:]
        return_dict['map_state'] = np.copy(self.map_state)
        return return_dict

    def frame_rendering(self):
        plt.clf()
        plt.title(
            "Turn {} - (m = {}, A = {}, d = {})".format(self.turns, self.metabolism, self.start_size, self.density))
        ax = plt.gca()

        cmap = colors.ListedColormap(["#000000", "#666666", "#90EE90", "#02FFFF"])
        bounds = [-1, 0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        x, y = np.meshgrid(list(range(100)), list(range(100)))
        plt.pcolormesh(
            x + 0.5,
            y + 0.5,
            np.transpose(self.map_state),
            cmap=cmap,
            norm=norm,
        )
        '''
        for x, y in state['bacteria']:
            plt.plot(
                x + 0.5,
                y + 0.5,
                color="black",
                marker="o",
                markersize=1,
                markeredgecolor="black",
            )
        '''
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        ax.set_aspect(1)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.invert_yaxis()

        msg = "In progress..."
        if self.amoeba_size >= self.goal_size:
            msg = "Goal size achieved!"
        elif self.turns == self.max_turns:
            msg = "Goal size not achieved."
        elif self.turns == 0:
            msg = "Starting state."

        cell_values = [["{}/{}".format(self.amoeba_size, self.goal_size)], [msg]]

        plt.table(
            cellText=cell_values,
            cellLoc='center',
            rowLabels=['Amoeba Size', 'Game State'],
            colLabels=[self.player_name],
        )
        plt.savefig("render/{}.png".format(self.turns))

        if self.use_gui:
            plt.pause(0.025)

    def frame_rendering_post(self):
        os.makedirs("render", exist_ok=True)

        old_files = glob("render/*.png")
        for f in old_files:
            os.remove(f)

        for i, state in enumerate(self.history):
            plt.clf()
            plt.title("Turn {} - (m = {}, A = {}, d = {})".format(i, self.metabolism, self.start_size, self.density))
            ax = plt.gca()

            cmap = colors.ListedColormap(["#000000", "#666666", "#90EE90", "#02FFFF"])
            bounds = [-1, 0, 1, 2, 3]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            x, y = np.meshgrid(list(range(100)), list(range(100)))
            plt.pcolormesh(
                x + 0.5,
                y + 0.5,
                np.transpose(state['map_state']),
                cmap=cmap,
                norm=norm,
            )
            '''
            for x, y in state['bacteria']:
                plt.plot(
                    x + 0.5,
                    y + 0.5,
                    color="black",
                    marker="o",
                    markersize=1,
                    markeredgecolor="black",
                )
            '''
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.xaxis.set_ticks_position("none")
            ax.yaxis.set_ticks_position("none")

            ax.set_aspect(1)
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 100])
            ax.invert_yaxis()

            msg = "In progress..."
            if state['amoeba_size'] >= self.goal_size:
                msg = "Goal size achieved!"
            elif i == self.max_turns:
                msg = "Goal size not achieved."
            elif i == 0:
                msg = "Starting state."

            cell_values = [["{}/{}".format(state['amoeba_size'], self.goal_size)], [msg]]

            plt.table(
                cellText=cell_values,
                cellLoc='center',
                rowLabels=['Amoeba Size', 'Game State'],
                colLabels=[self.player_name],
            )

            plt.savefig("render/{}.png".format(i))

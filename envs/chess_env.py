from math import sqrt

import gym
from gym import spaces
from time import sleep

import pygame
import numpy as np

def squared_euclidean_dist(a, b):
    return (a[0]-b[0])**2+(a[1]-b[1])**2

class RookCheckmateEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(self, render_mode=None, size=5, one_hot_observation_space=False, random_opponent=True, verbose=False):
        self.size = size
        self.random_opponent = random_opponent
        self.window_size = 512
        
        self.verbose = verbose
        
        self.one_hot_observation_space = one_hot_observation_space
        
        if one_hot_observation_space:
            self.num_squares = self.size**2
            self.observation_space = spaces.Dict(
                {
                    "wRook": spaces.Box(0, 1, shape=self.num_squares-1, dtype=int),
                    "wKing": spaces.Box(0, 1, shape=self.num_squares-1, dtype=int),
                    "bKing": spaces.Box(0, 1, shape=self.num_squares-1, dtype=int)
                })
        else:
            self.observation_space = spaces.Dict(
                {
                    "wRook": spaces.Box(0, size-1, shape=(2,), dtype=int),
                    "wKing": spaces.Box(0, size-1, shape=(2,), dtype=int),
                    "bKing": spaces.Box(0, size-1, shape=(2,), dtype=int)
                })
        
        # TODO this is a kludge.
        # Actions 0-7 are king moves 0=E, 1=NE, 2=N etc.
        # Actions 8-(size+6) are rook moves: 8=E, 9=EE, 10=EEE, etc.
        # Similarly, actions (size+7)-(2size+5) move North. West and South follow the same pattern
        self.action_space = spaces.Discrete(4*(size+1))
        
        self._action_to_direction = {
            0: ("wKing", np.array([1, 0])),
            1: ("wKing", np.array([1, 1])),
            2: ("wKing", np.array([0, 1])),
            3: ("wKing", np.array([-1, 1])),
            4: ("wKing", np.array([-1, 0])),
            5: ("wKing", np.array([-1, -1])),
            6: ("wKing", np.array([0, -1])),
            7: ("wKing", np.array([1, -1]))
        }
        
        idx = 8
        for direction in (np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])):
            for steps in range(1, size):
                self._action_to_direction[idx] = ("wRook", direction*steps)
                idx += 1
                
        self._pieces = {}
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # These will be None unless render_mode="human"
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        if self.one_hot_observation_space:
            out = {key : np.zeros(self.num_squares) for key in self._pieces.keys()}
            for key in out.keys():
                out[key][self._pieces[key][0]+self._pieces[key][1]*self.size] = 1
            return out
        else:
            return dict(self._pieces)
    
    def _get_info(self): return {}
    
    def _get_random_location(self):
        return self.np_random.integers(0, self.size, size=2, dtype=int)
    
    def _in_bounds(self, location):
        return min(location) >= 0 and max(location) < self.size
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._move_timer = 0
        
        self._pieces["wRook"] = self._get_random_location()
        self._pieces["wKing"] = self._get_random_location()
        while np.array_equal(self._pieces["wRook"], self._pieces["wKing"]):
            self._pieces["wKing"] = self._get_random_location()
        self._pieces["bKing"] = self._get_random_location()
        while self._is_threatened(self._pieces["bKing"]):
            self._pieces["bKing"] = self._get_random_location()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), self._get_info()
    
    def _is_threatened(self, coords):
        return ((abs(self._pieces["wKing"][0]-coords[0]) <= 1 and
            abs(self._pieces["wKing"][1]-coords[1]) <= 1) or
            self._pieces["wRook"][0] == coords[0] or
            self._pieces["wRook"][1] == coords[1])
    
    def _get_legal_bking_moves(self):
        moves = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                location = self._pieces["bKing"]+np.array([i, j])
                if self._in_bounds(location) and not self._is_threatened(location):
                    moves.append(location)
        return moves
        
    def _can_black_win(self):
        """
        Note that this assumes it is black to move, which is only true while step(action) is being processed.
        """
        # Check if black can take white king
        if (abs(self._pieces["wKing"][0]-self._pieces["bKing"][0]) <= 1 and
            abs(self._pieces["wKing"][1]-self._pieces["bKing"][1]) <= 1):
            return True
        # Check if black can take white rook without moving into check
        if (abs(self._pieces["wRook"][0]-self._pieces["bKing"][0]) <= 1 and
            abs(self._pieces["wRook"][1]-self._pieces["bKing"][1]) <= 1 and (
            abs(self._pieces["wKing"][0]-self._pieces["wRook"][0]) > 1 or
            abs(self._pieces["wKing"][1]-self._pieces["wRook"][1]) > 1)):
            return True
        # Black cannot win
        return False
    
    def _respond_to_invalid_move(self):
        return self._get_obs(), -100, True, False, self._get_info()
    
    def step(self, action):
        """
        Possible rewards:
        -1: Normal move
        -2: Illegal move. Move is not processed, so next observation will remain the same TODO is this the best way, or should the episode end with a large penalty?
        -100: Move that resulted in draw (e.g. throwing away the rook) or loss (e.g. move into check, rook sacrifice)
        100: Move that resulted in checkmate
        """
        piece, direction = self._action_to_direction[action]
        new_location = self._pieces[piece]+direction
        if not self._in_bounds(new_location):
            # If move is out-of-bounds, don't move and instead penalize with double the negative reward.
            if self.verbose:
                print(f"Invalid move (out-of-bounds) at turn {self._move_timer}")
            return self._respond_to_invalid_move()
        
        # TODO check for moving into check?
        if piece == "wRook": # Ensure we're not moving through or into another piece
            magnitude = max(np.absolute(direction))
            unit_move = direction/magnitude
            for i in range(magnitude):
                space = new_location-unit_move*i
                if i == 0 and np.array_equal(space, self._pieces["wKing"]):
                    if self.verbose:
                        print(f"Invalid move (into king) at turn {self._move_timer}")
                    return self._respond_to_invalid_move()
                elif any(np.array_equal(space, other_piece) for other_piece in self._pieces):
                    if self.verbose:
                        print(f"Invalid move (through piece) at turn {self._move_timer}")
                    return self._respond_to_invalid_move()
        elif piece == "wKing" and np.array_equal(new_location, self._pieces["wRook"]):
            if self.verbose:
                print(f"Invalid move (into rook) at turn {self._move_timer}")
            return self._respond_to_invalid_move()
        
        self._pieces[piece] = new_location # Execute move
        if self._can_black_win(): # If black can win, game is over
            if self.verbose:
                print(f"Blunder at turn {self._move_timer}")
            return self._get_obs(), -100, True, False, self._get_info()
        
        if self.render_mode == "human":
            sleep(0.2)
            self._render_frame()
        
        # Black moves (assuming he cannot win/draw this turn)
        opponent_moves = self._get_legal_bking_moves()
        if len(opponent_moves) == 0: # Black cannot move
            if self._is_threatened(self._pieces["bKing"]): # Black is in checkmate
                if self.verbose:
                    print(f"Checkmate! at turn {self._move_timer}")
                return self._get_obs(), 100, True, False, self._get_info()
            else: # Black is in stalemate
                if self.verbose:
                    print(f"Stalemate at turn {self._move_timer}")
                return self._get_obs(), -100, True, False, self._get_info()
        
        # Black must have at least one legal move.
        if self.random_opponent:
            # Choose one at random.
            move = self.np_random.choice(opponent_moves)
        else:
            # Make the move that minimizes the (eucledian!) distance to the enemy rook. In case of a tie, prefer the center of the board.
            move_options = []
            best_dist = float("inf")
            for move in opponent_moves:
                dist = squared_euclidean_dist(move, self._pieces["wRook"])
                if dist < best_dist:
                    move_options = [move,]
                    best_dist = dist
                elif dist == best_dist:
                    move_options.append(move)
            if len(move_options) == 1:
                move = move_options[0]
            else:
                move_options_center = []
                center_pos = ((self.size-1)/2, (self.size-1)/2)
                best_dist = float("inf")
                for move in move_options:
                    dist = squared_euclidean_dist(move, center_pos)
                    if dist < best_dist:
                        move_options_center = [move,]
                        best_dist = dist
                    elif dist == best_dist:
                        move_options_center.append(move)
                move = self.np_random.choice(move_options_center) # This will be a 1-element list unless there are two legal moves that are equidistant from both the center and the enemy rook.
                
        self._pieces["bKing"] = move
        
        if self.render_mode == "human":
            self._render_frame()
        
        self._move_timer += 1
        if self._move_timer == 50:
            if self.verbose:
                print("50-move rule (timeout)")
            return self._respond_to_invalid_move()
        
        return self._get_obs(), -1, False, False, self._get_info()
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((127, 127, 127))
        square_size = self.window_size // self.size
        
        # Draw white rook
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            pygame.Rect(
                square_size * (self._pieces["wRook"]+1/8),
                (square_size*0.75, square_size*0.75)
            )
        )
        
        # Draw white king
        pygame.draw.circle(
            canvas,
            (255, 255, 255),
            (self._pieces["wKing"]+0.5)*square_size,
            square_size*3/8
        )
        
        # Draw black king
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self._pieces["bKing"]+0.5)*square_size,
            square_size*3/8
        )
        
        for i in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, square_size*i), (self.window_size, square_size*i), width=3)
            pygame.draw.line(canvas, 0, (square_size*i, 0), (square_size*i, self.window_size), width=3)
        
        # TODO checkerboard pattern?
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes(1, 0, 2))
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    pass

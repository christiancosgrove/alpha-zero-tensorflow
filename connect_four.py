import numpy as np
import copy
import random
import time
import os
import pickle
from model import Model

""" MCTS Node. Contains information about expected game value, policy probabilities, and visit counts. """
class Node:
    def __init__(self, board, parent=None, action=None, prob=None):
        self.parent = parent
        # action taken to get to this node
        self.action = action
        self.num_visits = 0
        self.total_value = 0
        self.average_value = 0
        self.children = {}
        # policy probabilities for all children
        # cache if multiple children get expanded - can reuse
        self.children_probs = None
        # policy probability for this move
        self.prob = prob
        self.over = board.is_game_over()
        self.possible_actions = board.moves()

    """ In the selection (tree-traversal) phase of MCTS, choose the action that yields the largest selection_criterion """
    def selection_criterion(self, turn_parity, t, N):
        p = self.prob if self.prob is not None else 0
        s = turn_parity * self.average_value + 1.0 * np.sqrt(N) * p / (1 + self.num_visits)
        return s

    """ For use in the selection phase of MCTS """
    def select(self, turn_parity, t):
        N = sum(self.children[i].num_visits for i in self.children)
        return self.children[max(self.children, key=lambda i: self.children[i].selection_criterion(turn_parity, t, N))]

    """ For use in the expansion phase of MCTS """
    def add_child(self, action, board, prob):
        self.children[action] = Node(board, parent=self, action=action, prob=prob)
        return self.children[action]

    """ For use in the backpropagation phase of MCTS """
    def update(self, value):
        self.total_value += value
        self.num_visits += 1
        self.average_value = self.total_value / self.num_visits

""" Connect Four Board with AlphaZero/MCTS movemaking routines. """
class Board:
    def __init__(self):
        self.width = 7
        self.height = 6

        self.state = np.zeros(shape=(self.width, self.height), dtype=np.int64)
        self.turn = 0

        self.root = None
        self.board_probabilities = {}

    """ If possible, perform a move in the move space. If not possible, return -1. """
    def make_move(self, action):
        h = 0
        while self.state[action, h] != 0:
            h += 1
            if h >= self.height:
                return -1

        self.state[action, h] = self.get_player_piece()
        if self.root is not None:
            if action in self.root.children:
                self.root = self.root.children[action]
                self.root.parent = None
            else:
                self.root = None
        self.turn += 1

    def moves(self):
        if self.is_game_over() is not None:
            return []
        return list(filter(lambda i: self.state[i, -1] == 0, list(range(self.width))))

    """ Check if the game is over. If so, return 1 if player 1 wins, -1 if player -1 wins, and 0 if a draw occurs.
    Otherwise, return None."""
    def is_game_over(self):
        # only need to check current player

        empty = 0
        opponent_piece = self.get_opponent_player_piece()

        #horizontal
        for y in range(self.height):
            count = 0
            for x in range(self.width):
                if self.state[x, y] == 0:
                    empty += 1

                if self.state[x, y] != opponent_piece:
                    count = 0
                else:
                    count += 1

                if count == 4:
                    return opponent_piece
        
        # vertical
        for x in range(self.width):
            count = 0
            for y in range(self.height):
                if self.state[x, y] != opponent_piece:
                    count = 0
                else:
                    count += 1
                if count == 4:
                    return opponent_piece

        # diagonal, positive slope
        for x in range(-self.height, self.width):
            count = 0
            for y in range(self.height):
                if x + y >= self.width or x + y < 0:
                    continue
                if self.state[x + y, y] != opponent_piece:
                    count = 0
                else:
                    count += 1
                if count == 4:
                    return opponent_piece

        # diagonal, negative slope
        for x in range(0, self.width + self.height):
            count = 0
            for y in range(self.height):
                if x - y >= self.width or x - y < 0:
                    continue
                if self.state[x - y, y] != opponent_piece:
                    count = 0
                else:
                    count += 1
                if count == 4:
                    return opponent_piece
        if empty == 0:
            return 0

        return None

    def get_player_piece(self):
        if self.turn % 2 == 0:
            return 1
        else:
            return -1

    def get_opponent_player_piece(self):
        if self.turn % 2 == 0:
            return -1
        else:
            return 1

    def print(self):
        print(np.transpose(self.state[:,::-1]))

    """ Perform a single iteration of MCTS. Involves first selecting for the most promising sequence of moves, 
        expanding the game tree by a single node, and backpropagating the value of that state (using either the game
        result or the value head of the model) """ 
    def step(self, model, num_iterations=64, reuse_root=False, noise=False):
        if self.root is None:
            self.root = Node(board=self)

        t = 0
        while self.root.num_visits < num_iterations:
            node = self.root
            board = copy.deepcopy(self)

            while len(node.possible_actions) == 0 and len(node.children) > 0:
                node = node.select(board.get_player_piece(), t)
                board.make_move(node.action)
                t+=1
            
            if node.over is None:
                if node.children_probs is None:
                    probs, _ = model.predict(np.expand_dims(board.training_state(), axis=0))
                    node.children_probs = probs
                probs_possible = node.children_probs[0, np.array(node.possible_actions)]
                epsilon = 0.2
                probs_possible = (1.0 - epsilon) * probs_possible + epsilon * np.random.dirichlet([0.8] * probs_possible.shape[0])
                probs_possible /= np.sum(probs_possible)
                idx = np.random.choice(np.arange(len(node.possible_actions)), p=probs_possible)
                action = node.possible_actions[idx]
                del node.possible_actions[idx]
                board.make_move(action)
                node = node.add_child(action, board, np.float(node.children_probs[0, idx]))
            
            # MCTS Rollout
            # while board.is_game_over() is None:
            #     prob, _ = model.predict(np.expand_dims(board.training_state(), axis=0))
            #     prob = prob[0]
            #     board.make_move(np.random.choice(self.width))#, p=prob))

            value = board.is_game_over()
            if value is None:
                policy_probs, value = model.predict(np.expand_dims(board.training_state(), axis=0))
                node.children_probs = policy_probs
                value *= board.get_player_piece()
            while node is not None:
                node.update(value)
                node = node.parent


        total_visits = sum(self.root.children[i].num_visits for i in self.root.children)
        probs = np.zeros(shape=self.width, dtype=np.float32)
        for i in self.root.children:
            probs[i] = self.root.children[i].num_visits / total_visits
        if not reuse_root:
            self.root = None
        return probs

    def play(self, model):
        self.make_move(np.argmax(self.step(model, num_iterations=200)))

    """ Tensor state to be passed to NN model """
    def training_state(self):
        s = self.get_player_piece() * self.state
        white = s.astype(np.float32)
        white[white < 0] = 0
        black = s.astype(np.float32)
        black[black > 0] = 0
        black *= -1
        # like AlphaZero paper: use an entire channel to indicate who's the current player
        # for simplicity, we model connect4 as a symmetric game
        # therefore, we can just flip all the pieces fed into the evaluation/policy NN
        # turn = np.full_like(white, self.get_player_piece()) 
        return np.array([white, black]).transpose([1,2,0])

    """ Perform a single game of self-play and return intermediate game states, MCTS probs, and the game result."""
    def self_play_game(self, model):
        states = []
        probs = []
        players = []
        while self.is_game_over() is None:
            states.append(self.training_state())
            players.append(self.get_player_piece())
            p = self.step(model, reuse_root=True, noise=True)
            self.print()
            p/=np.sum(p)
            probs.append(p)
            if self.turn > 10:
                self.make_move(np.argmax(p))
            else:
                self.make_move(np.random.choice(p.shape[0], p=p))

        game_over = self.is_game_over()
        if game_over == 1:
            print('white wins!')
        elif game_over == -1:
            print('black wins!')
        else:
            print('draw!')
        value = np.float(self.is_game_over())
        return np.array(states), np.array(probs), value * np.expand_dims(np.array(players), axis=1)

    """ Save training data to disk. """
    def save_game(self, model, data_dir='data'):
        with open(os.path.join(data_dir, str(time.time()) + '.pkl'), 'wb') as file:
            pickle.dump(self.self_play_game(model), file)
    
    """ Play two models against each other for evaluation purposes. 
        Returns a positive number if model2 beats model1 on average, negative if model1 beats model2. """
    def compare_models(self, model, model1, model2, games=30):
        value = 0
        model = Model(self.width, self.height)
        for i in range(games):
            board = Board()
            # TODO: load two models simultaneously while playing one model against another
            # requires some tensorflow magic
            while board.is_game_over() is None:
                name = model1 if (i + self.turn) % 2 == 0 else model2
                model.restore_from_file(name)
                p = board.step(model, reuse_root=False)
                # introduce some randomness into move-making process to get a 
                # diverse comparison of the two models
                board.make_move(np.random.choice(p.shape[0], p=p))
            win = board.is_game_over()
            if i % 2 == 0:
                win *= -1
            print('win',win)
            value += board.is_game_over()
        return value / games
        


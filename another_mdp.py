"""Need another MDP."""

from mdp import MarkovDecisionProcess
from itertools import product, permutations, combinations
import numpy as np
from copy import deepcopy

def get_indices_where(b):
    nonzero_values = np.nonzero(b)
    return [(x, y) for x, y in zip(*nonzero_values)]

def valid_edge(p1, p2):
    # at least one dimension must match
    row_diff = abs(p1[0] - p2[0])
    col_diff = abs(p1[1] - p2[1])
    if row_diff > 1 or col_diff > 1:
        return False
    only_one_match = (row_diff == 1) != (col_diff == 1)
    return only_one_match

def valid_snake(s):
    board, direction = s
    
    row, col = get_indices_where(board == 2)[0]
    
    ones = get_indices_where(board == 1)
    if len(ones) == 0:
        return True
    
    if direction == 'right':
        next_one_location = row, col - 1
    elif direction == 'up':
        next_one_location = row + 1, col
    elif direction == 'down':
        next_one_location = row - 1, col
    elif direction == 'left':
        next_one_location = row, col + 1
    else:
        raise Exception('Unexpected case.')
        
    if next_one_location not in ones:
        return False
    
    
    edges = {}
    for p1 in ones:
        edges[p1] = []
        for p2 in ones:
            if valid_edge(p1, p2) and p2 != next_one_location:
                edges[p1].append(p2)

    

    best_path = longest_path(
        path=[next_one_location],
        available_edges=deepcopy(edges)
    )
    # only if we can traverse each point does this board count
    return len(best_path) == len(ones)

def longest_path(path, available_edges):
    latest_point = path[-1]
    next_possible_points = available_edges[latest_point]
    if len(next_possible_points) == 0:
        return path
    options = {p: None for p in next_possible_points}
    for next_point in options:
        possible_path = deepcopy(path)
        possible_path.append(next_point)
        hypothetical_remaining_edges = {
            p: available_edges[p]
            for p in available_edges
        }.copy()
        for hypo_edge_key in hypothetical_remaining_edges:
            if next_point in hypothetical_remaining_edges[hypo_edge_key]:
                hypothetical_remaining_edges[hypo_edge_key].remove(next_point)
        options[next_point] = longest_path(
            possible_path,
            deepcopy(hypothetical_remaining_edges)
        )
    best_length = max(len(options[p]) for p in options)
    best_path = [
        options[p] for p in options
        if len(options[p]) == best_length
    ][0]
    return best_path

def valid_state(s):
    # board composed of 0s, 1, 2, 3
    board, direction = s
    flattened = board.flatten()
    if flattened[(flattened > 3) | (flattened < 0)].shape != (0,):
        return False
    if flattened[flattened == 2].shape != (1,):
        return False
    if flattened[flattened == 3].shape != (1,):
        return False
    # now the tricky part, checking that a string of 1s exists
    return valid_snake(s)

def get_states(board_length, max_snake_length, max_state_space, verbosity_rate=1000):
    directions = ['up', 'down', 'left', 'right']
    board_size = board_length ** 2
    index_options = range(board_length)
    all_indices = list(product(index_options, index_options))
    
    states = []
    for i in range(1, max_snake_length + 1):
        new_choices = combinations(all_indices, i)
        print(
            f'Trying max snake length {i} with '
            f'{len(list(combinations(all_indices, i))):,.0f} base combos'
        )
        for choice in new_choices:
            board = np.zeros(shape=(board_length, board_length))
            for one_loc in choice:
                board[one_loc] = 1
            
            one_locations = np.nonzero(board == 1)
            for row_1, col_1 in zip(*one_locations):
                board_copy = board.copy()
                board_copy[row_1, col_1] = 2
            
                zero_locations = np.nonzero(board == 0)
                for row_0, col_0 in zip(*zero_locations):
                    board_copy_copy = board_copy.copy()
                    board_copy_copy[row_0, col_0] = 3
                            
                    for d in directions:
                        candidate_state = (board_copy_copy, d)
                        if valid_state(candidate_state):
                            states.append(candidate_state)
                            if len(states) % verbosity_rate == 0:
                                print(f'{len(states):,.0f} states found...')
                            if len(states) >= max_state_space:
                                print(f'i: {i}')
                                return states
    return states

def hashable_state(s):
    board, direction = s
    return (tuple(board.reshape(-1)), direction)

def array_state(s_hashable):
    tuple_board, direction = s_hashable
    board_length = int(len(tuple_board) ** 0.5)
    board = np.array(tuple_board).reshape((board_length, board_length))
    return (board, direction)

class SnakeMDP(MarkovDecisionProcess):
    
    def __init__(self, board_length=5, max_snake_length=4, max_state_space=5e4):
        board_states = get_states(board_length=board_length, max_snake_length=max_snake_length, max_state_space=max_state_space)
        
        # need to convert these states into something hashable
        self.states = [hashable_state(s) for s in board_states]
        
        states = product(
            [np.zeros(shape=(board_length, board_length))],
            self.directions
        )
        
        
              
        
    def transition_model(self, s, a, s_prime):
        pass
    
    def reward(self, s):
        pass
    
    def accessible_states(self, s):
        pass
    
    def available_actions(self, s):
        pass
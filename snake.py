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

def states_longest_path_of_ones(s):
    board, direction = s
    
    row, col = get_indices_where(board == 2)[0]
    
    ones = get_indices_where(board == 1)
    if len(ones) == 0:
        return [], []
    
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
        raise ValueError
    
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
    return best_path, ones
    
def valid_snake(s):
    try:
        best_path, ones = states_longest_path_of_ones(s)
    except ValueError:
        return False
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
    for i in range(1, max_snake_length): # don't want to actually have max snake length be equal to i at any point
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
    
    def __init__(self, win_value=1, lose_value=-1, exist_factor=0.1,
                 board_length=5, max_snake_length=4, max_state_space=5e4):
        self.WIN = win_value
        self.LOSE = lose_value
        self.exist_factor = exist_factor # for each turn you survive, multiply the number of 1s you've accumulated by this factor as a reward
        self.max_snake_length = max_snake_length
        self.board_length = board_length
        
        board_states = get_states(board_length=board_length, max_snake_length=max_snake_length, max_state_space=max_state_space)
        
        # need to convert these states into something hashable
        self.states = [hashable_state(s) for s in board_states]
        self.actions = ['up', 'down', 'left', 'right']
        
        # add 8 extra states to the list of possible states to capture the OOB locations
        for a in self.actions:
            self.states.append((None, a))
            self.states.append(('WIN', a))
        
    def transition_model(self, s, a, s_prime):
        accessibles = self.accessible_states(s, a)
        assert s_prime in accessibles
        return 1.0 / len(accessibles)
    
    def reward(self, s):
        if s[0] is None:
            return self.LOSE
        if s[0] == 'WIN':
            return self.WIN
        best_one_path, ones = states_longest_path_of_ones(
            array_state(s)
        )
        return len(best_one_path) * self.exist_factor
    
    def accessible_states(self, s, a):
        if s[0] is None:
            return []
        if s[0] == 'WIN':
            return []
        s_array = array_state(s)
        board, _ = s_array
        best_one_path, ones = states_longest_path_of_ones(
            s_array
        )
        current_location_of_2 = get_indices_where(board == 2)[0]
        
        new_direction = a
        new_array = board.copy()
        
        row, col = current_location_of_2
        
        if new_direction == 'right':
            next_two_location = row, col + 1
        elif new_direction == 'up':
            next_two_location = row - 1, col
        elif new_direction == 'down':
            next_two_location = row + 1, col
        elif new_direction == 'left':
            next_two_location = row, col - 1
        else:
            raise Exception('Unexpected case.')
            
        # case 0-a: are we heading into the OOB field?
        if not (0 <= next_two_location[0] < self.board_length) \
            or not (0 <= next_two_location[1] < self.board_length): # check this
            return [(None, a)]
        
        # case 0-b: are we heading into an existing 1?
        if board[next_two_location] == 1:
            return [(None, a)]
        
        # case 1: 2 is heading into a 0-spot
        if board[next_two_location] == 0:
            if len(best_one_path) > 0:
                last_1_location = best_one_path[-1]
                new_array[last_1_location] = 0
                new_array[current_location_of_2] = 1
            else:
                new_array[current_location_of_2] = 0
            new_array[next_two_location] = 2
            return [hashable_state((new_array, new_direction))]
            
        # case 2: 2 is heading into a 3-spot
        if board[next_two_location] == 3:
            # may send to terminal WIN node if the snake is long enough
            if len(best_one_path) + 2 == self.max_snake_length:
                return [('WIN', new_direction)]
            new_array[next_two_location] = 2
            new_array[current_location_of_2] = 1
            
            # need to sample possible locations for new 3
            possibilities = []
            zero_locations = np.nonzero(new_array == 0)
            for row_0, col_0 in zip(*zero_locations):
                new_array_copy = new_array.copy()
                new_array_copy[row_0, col_0] = 3
                possibilities.append(hashable_state((new_array_copy, new_direction)))
            return possibilities
        
        # didn't expect this case
        raise Exception('Whoa...')
    
    def available_actions(self, s):
        _, direction = s
        if direction == 'up': opposite = 'down'
        elif direction == 'down': opposite = 'up'
        elif direction == 'left': opposite = 'right'
        elif direction == 'right': opposite = 'left'
        else: raise Exception('Unexpected...')
        
        actions = deepcopy(self.actions)
        actions.remove(opposite)
        return actions
        
        
        

example_1 = hashable_state(
    (np.array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [3., 1., 1., 1., 2.]]),
     'right')
)

example_2 = hashable_state(
    (np.array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 3., 0.],
               [0., 0., 0., 2., 0.],
               [0., 1., 1., 1., 0.]]),
     'up')
)
s = example_2

mdp = SnakeMDP(board_length=5, max_snake_length=7, max_state_space=2e5)


snake_policy, snake_value = mdp.policy_iteration(gamma=0.5, epsilon=0.0001, max_allowed_time=720)

# s = ((1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'down')
# array_state(s)
# array_state(mdp.accessible_states(s, 'down')[0])

for s in random.sample(mdp.states, k=5):
    print(f'\nFor state s = \n{array_state(s)}')
    print(f'the recommended action is: {snake_policy[s]}')

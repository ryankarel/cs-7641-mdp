"""Set up blackjack problem."""

import numpy as np
import pandas as pd
from itertools import product
import random
import time

from copy import copy

LOSE = -1
WIN = 1

DEALER_STAND_THRESHOLD = 17
MAX_HITTABLE_VALUE = 21

verbose = False



class MarkovDecisionProcess:

    def __init__(self, states, actions, transition_model, reward, accessible_states):
        self.states = states # a list
        self.actions = actions # a list
        self.transition_model = transition_model # a callable
        self.reward = reward # a callable
        self.accessible_states = accessible_states # a callable
        self.available_actions = None # a callable
        
    def _sample_state_layout(self):
        s = random.choice(self.states)
        print(f'Randomly sampled state: {s}')
        available_actions = self.available_actions(s)
        print(f'Available actions: {available_actions}')
        a = random.choice(available_actions)
        print(f'Available states from s={s} and a={a}:')
        probs = []
        for s_prime in self.accessible_states(s, a):
            trans_prob = self.transition_model(s,a,s_prime)
            probs.append(trans_prob)
            print(f'\ts\'={s_prime}; T(s, a, s\')={trans_prob:.3f}; R(s\')={self.reward(s_prime):.3f}')
        print(f'Total sum of probabilities: {sum(probs):.3f}')

    def policy_iteration(self, gamma, random_state=0, max_allowed_time=60, max_iter=100):
        '''Solve MDP with policy iteration.
        
        https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node20.html
        '''
        start = time.time()
        # randomly initialize value function
        V = dict(zip(
            self.states,
            [0] * len(self.states)
        ))

        pi = {
            s: random.choice(self.available_actions(s))
            for s in self.states
        }
        
        iteration = 0

        terminate_algorithm = False
        while not terminate_algorithm:
            iteration += 1
            if time.time() - start > max_allowed_time:
                print('Time ran out!')
                break
            if iteration > max_iter:
                print('Too many iterations.')
                break
            # policy evaluation
            for s in self.states:
                v = V[s]
                recommended_action = pi[s]
                V[s] = self.reward(s) + gamma * sum(
                    self.transition_model(s, recommended_action, s_prime) * V[s_prime]
                    for s_prime in self.accessible_states(s, recommended_action)
                )
            
            # policy improvement
            policy_stable = True
            changed_values = 0
            for s in self.states:
                available_actions = self.available_actions(s)
                if len(available_actions) == 1:
                    pi[s] = available_actions[0]
                    continue
                action_values = pd.Series(index=available_actions, dtype=float)
                for a in available_actions:
                    action_values.loc[a] = self.reward(s) + gamma * sum(
                        self.transition_model(s, a, s_prime) * V[s_prime]
                        for s_prime in self.accessible_states(s, a)
                    )
                assert action_values.notnull().all()
                new_recommended_action = action_values.idxmax()
                if new_recommended_action != pi[s]:
                    policy_stable = False
                    changed_values += 1
                pi[s] = new_recommended_action
            print(f'At iteration {iteration}, changed policy values for {changed_values} states.')
                
            if policy_stable:
                terminate_algorithm = True
                
        return pi

    def value_iteration(self, gamma, epsilon=0.001, random_state=0, max_allowed_time=60, max_iter=100):
        start = time.time()
        # randomly initialize value function
        V = dict(zip(
            self.states,
            [0] * len(self.states)
        ))

        iteration = 0

        terminate_algorithm = False
        while not terminate_algorithm:
            max_change_in_value = 0
            iteration += 1
            if time.time() - start > max_allowed_time:
                print('Time ran out!')
                break
            if iteration > max_iter:
                print('Too many iterations.')
                break
            
            value_change = []
            
            for s in self.states:
                proposed_value = max(
                    self.reward(s) + gamma * sum(
                        self.transition_model(s, a, s_prime) * V[s_prime]
                        for s_prime in self.accessible_states(s, a)
                    )
                    for a in self.available_actions(s)
                )
                abs_value_change = abs(proposed_value - V[s])
                value_change.append(abs_value_change)
                max_change_in_value = max(max_change_in_value, abs_value_change)
                V[s] = proposed_value
                
            avg_value_change = np.average(value_change)
            print(f'At iteration {iteration}, max change in value: {max_change_in_value:.5f}; avg. change: {avg_value_change:.5f}')
            if max_change_in_value < epsilon:
                break
            
        pi = {}
            
        for s in self.states:
            available_actions = self.available_actions(s)
            if len(available_actions) == 1:
                pi[s] = available_actions[0]
                continue
            action_values = pd.Series(index=available_actions, dtype=float)
            for a in available_actions:
                action_values.loc[a] = self.reward(s) + gamma * sum(
                    self.transition_model(s, a, s_prime) * V[s_prime]
                    for s_prime in self.accessible_states(s, a)
                )
            assert action_values.notnull().all()
            pi[s] = action_values.idxmax()
                
        return pi
        

class BlackjackMDP(MarkovDecisionProcess):
    '''Blackjack Markov Decision Process.
    
    Assume that aces are always worth eleven.
    
    '''

    def __init__(self):
        self.states = list(product(
            range(2, 33), # total point value of player's hand
            range(2, 33), # what value shows for the dealer
            # range(0, 22), # player aces
            # range(0, 22), # dealer aces
            ['hitting', 'stand'] # player state
        ))
        
        for any_point_value in range(2, 33):
            for hitted_dealer_value in range(12, 33):
                for player_ace in range(0, 22):
                    for dealer_ace in range(0, 22):
                        # we will never see this case if the player is still hitting
                        self.states.remove((any_point_value, hitted_dealer_value, player_ace, dealer_ace, 'hitting'))
        
        self.actions = ['hold', 'hit'] # to hit or not hit
        
    def _standing(self, s):
        return s[-1] == 'stand'
    
    def _player_aces(self, s):
        return s[2]
    
    def _dealer_aces(self, s):
        return s[3]
    
    def _player_point_value(self, s):
        return s[0]
    
    def _dealer_point_value(self, s):
        return s[1]
    
    def available_actions(self, s):
        if self._player_point_value(s) > 21 or self._standing(s):
            return ['hold']
        else:
            return ['hold', 'hit']

    def accessible_states(self, s, a):
        assert a in self.available_actions(s)
        assert s in self.states
        
        possibilities = []
        
        if self._dealer_point_value(s) >= DEALER_STAND_THRESHOLD or self._player_point_value(s) > MAX_HITTABLE_VALUE:
            # terminal state reached
            return []
        
        if a == 'hold' and not self._standing(s):
            return [ s[:-1] + ('stand',) ] # only one option from here
        
        if self._standing(s):
            # dealer must be below threshold by this point
            for additional_points in range(2, 12):
                s_prime = list(copy(s))
                s_prime[1] += additional_points
                possibilities.append(tuple(s_prime))
            return possibilities
        
        assert a == 'hit'
        
        for additional_points in range(2, 12):
            s_prime = list(copy(s))
            s_prime[0] += additional_points
            possibilities.append(tuple(s_prime))
        return possibilities
        
    def transition_model(self, s, a, s_prime):
        accessibles = self.accessible_states(s, a)
        assert s_prime in accessibles
        if len(accessibles) == 1:
            return 1.0
        
        if a == 'hit':
            value_change = self._player_point_value(s_prime) - self._player_point_value(s)
            assert value_change > 0
            
            if (2 <= value_change <= 9) or value_change == 11:
                return 1 / 13
            elif value_change == 10:
                return 4 / 13
        
        if a == 'hold':
            value_change = self._dealer_point_value(s_prime) - self._dealer_point_value(s)
            assert value_change > 0
            
            if (2 <= value_change <= 9) or value_change == 11:
                return 1 / 13
            elif value_change == 10:
                return 4 / 13
            
        raise Exception('Unexpected case')

    def reward(self, s):
        if self._player_point_value(s) > 21:
            return LOSE
        if not self._standing(s) or self._dealer_point_value(s) < DEALER_STAND_THRESHOLD:
            # here we are not in stand mode, which means no reward yet
            return 0
        if self._player_point_value(s) == 21:
            return WIN
        if self._dealer_point_value(s) > 21:
            return WIN
        
        if self._player_point_value(s) > self._dealer_point_value(s):
            return WIN
        return LOSE
        

mdp = BlackjackMDP()


mdp._sample_state_layout()

for s in mdp.accessible_states((11, 3, 'hitting'), 'hit'):
    print(s, mdp.reward(s))
mdp.transition_model((19, 6, 'hitting'), 'hold', (19, 8, 'hitting'))
blackjack_policy = mdp.policy_iteration(0.99)
blackjack_policy = mdp.value_iteration(0.05, max_allowed_time=180)

policy_visualization = pd.DataFrame(
    index=pd.Index(range(2, 12), name='Dealer Value'),
    columns=pd.Index(range(2, 22), name='Player Value'),
    dtype='string'
)
for i in range(2, 12):
    for j in range(2, 22):
        policy_visualization.loc[i, j] = blackjack_policy[(j, i, 'hitting')]

policy_visualization.iloc[:, 5:15]

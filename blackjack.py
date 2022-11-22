"""Set up blackjack problem."""

import numpy as np
import pandas as pd
from itertools import product
import random
import time

from copy import copy

LOSE = -1
WIN = 1

verbose = False


def deal():
    probabilities = np.array([1.0] * 8 + [4.0] + [1.0])
    probabilities /= probabilities.sum()
    random_card_value = np.random.choice(
        range(2, 12),
        p=probabilities
    )
    return random_card_value


def dealer_hit(cards):
    if verbose: print(f'Dealer cards: {cards}')
    return sum(cards) < 17


def bust(cards):
    return sum(cards) > 21


def hand_value(cards):
    face_value = sum(cards)
    return face_value
        

def compare_hands(player_cards, dealer_cards):
    if verbose:
        print(f'Player has {player_cards}, dealer has {dealer_cards}')
    if hand_value(player_cards) > hand_value(dealer_cards):
        if verbose: print('Player wins')
        return WIN
    else:
        if verbose: print('Player loses')
        return LOSE


class MarkovDecisionProcess:

    def __init__(self, states, actions, transition_model, reward, accessible_states):
        self.states = states # a list
        self.actions = actions # a list
        self.transition_model = transition_model # a callable
        self.reward = reward # a callable
        self.accessible_states = accessible_states # a callable

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

        pi = dict(zip(
            self.states,
            [random.choice(self.actions)] * len(self.states)
        ))
        
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
                action_values = pd.Series(index=self.actions, dtype=float)
                for a in self.actions:
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
                    for a in self.actions
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
            action_values = pd.Series(index=self.actions, dtype=float)
            for a in self.actions:
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
            range(2, 33, 1), # total point value of player's hand
            range(2, 12), # what value shows for the dealer
            ['hitting', 'stand'] # player state
        ))

        self.actions = ['hold', 'hit'] # to hit or not hit
        
    def _get_player_state(self, s):
        return s[-1]
    
    def _point_value(self, s):
        return s[0]

    def accessible_states(self, s, a):
        assert a in self.actions
        assert s in self.states
        
        if self._get_player_state(s) == 'stand':
            return []
        
        if a == 'hold' or self._point_value(s) >= 21:
            return [ s[:2] + ('stand',) ] # only one option from here
        
        assert a == 'hit'
        
        possibilities = []
        
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
            value_change = self._point_value(s_prime) - self._point_value(s)
            assert value_change > 0
            
            if (2 <= value_change <= 9) or value_change == 11:
                return 1 / 13
            elif value_change == 10:
                return 4 / 13
            
        raise Exception('Unexpected case')

    def reward(self, s):
        if self._get_player_state(s) == 'hitting':
            # here we are not in stand mode, which means no reward yet
            return 0

        if self._point_value(s) > 21:
            return LOSE
        if self._point_value(s) == 21:
            return WIN
        
        reward_sequence = []
        
        for _ in range(2000):
            dealer_cards = [s[1], deal()]
            while dealer_hit(dealer_cards):
                dealer_cards.append(deal())
                if bust(dealer_cards):
                    reward_sequence.append(WIN)
            if self._point_value(s) >= hand_value(dealer_cards):
                reward_sequence.append(WIN)
            else:
                reward_sequence.append(LOSE)
        
        return np.average(reward_sequence)

mdp = BlackjackMDP()
# pi_star = mdp.policy_iteration(0.04)
# pi_star = mdp.value_iteration(0.05, max_allowed_time=180)



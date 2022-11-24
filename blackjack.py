"""Set up blackjack problem."""

import numpy as np
import pandas as pd
from itertools import product
import random
import time
from mdp import MarkovDecisionProcess

from copy import copy

LOSE = -1
WIN = 1

DEALER_STAND_THRESHOLD = 17
MAX_HITTABLE_VALUE = 21


class SimplifiedBlackjackMDP(MarkovDecisionProcess):
    '''Blackjack Markov Decision Process.
    
    Assume that aces are always worth eleven.
    
    '''

    def __init__(self):
        self.states = list(product(
            range(2, 33), # total point value of player's hand
            range(2, 33), # what value shows for the dealer
            ['hitting', 'stand'] # player state
        ))
        
        for any_point_value in range(2, 33):
            for hitted_dealer_value in range(12, 33):
                # we will never see this case if the player is still hitting
                self.states.remove((any_point_value, hitted_dealer_value, 'hitting'))
        
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
        

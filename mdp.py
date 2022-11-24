# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:33:13 2022

@author: rache
"""

import random
import numpy as np
import time
import pandas as pd
from copy import deepcopy

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

    def policy_iteration(self, gamma, epsilon=0.001, random_state=0, max_allowed_time=60, max_iter=100):
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
            max_change_in_value = epsilon + 1
            while max_change_in_value > epsilon:
                max_change_in_value = 0
                for s in self.states:
                    recommended_action = pi[s]
                    try:
                        proposed = self.reward(s) + gamma * sum(
                            self.transition_model(s, recommended_action, s_prime) * V[s_prime]
                            for s_prime in self.accessible_states(s, recommended_action)
                        )
                    except Exception:
                        print(s, recommended_action)
                        raise
                    max_change_in_value = max(max_change_in_value, abs(proposed - V[s]))
                    V[s] = proposed
                
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
                
        return pi, V

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
                
        return pi, V
        
    def Q_learning(self, gamma, epsilon=0.001, random_state=0, max_allowed_time=60, max_iter=100):
        Q = {
            (s, a): 0
            for s in self.states
            for a in self.accessible_states(s)
        }
        pi = {
            s: random.choice(self.available_actions(s))
            for s in self.states
        }
        
        Q_copy = deepcopy(Q)
        terminate_algorithm = False
        max_change_in_value = 0
        while not terminate_algorithm:
            max_change_in_value = 0
            for s, a in Q:
                # sample new state s'
                possible_s_prime = self.accessible_states(s)
                probabilities = [
                    self.transition_model(s, a, s_prime)
                    for s_prime in possible_s_prime
                ]
                s_prime = random.choices(possible_s_prime, probabilities)[0]
                proposed_value = self.reward(s) + gamma * max(
                    Q_copy[(s_prime, a_prime)]
                    for a_prime in self.available_actions(s_prime)
                )
                max_change_in_value = max(
                    max_change_in_value,
                    abs(Q[(s, a)] - proposed_value)
                )
                Q[(s, a)] = proposed_value
                Q_copy = deepcopy(Q)
            if max_change_in_value < epsilon:
                terminate_algorithm = True
                
        for s in pi:
            Q_a = {
                a: Q[(s, a)]
                for a in self.available_actions(s)
            }
            pi[s] = max(Q_a, key=Q_a.get)
            
        return pi, Q
                
        
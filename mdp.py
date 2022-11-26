

import random
import numpy as np
import time
import pandas as pd
from copy import deepcopy

def initialize_statistics():
    return pd.DataFrame(index=pd.Index([], name='Iteration'), columns=['Time (s)', 'Max Changes'])
    
def update_stats(data, iteration, time_delta, max_value_change):
    data.loc[iteration, 'Time (s)'] = time_delta
    data.loc[iteration, 'Max Changes'] = max_value_change
    data = data.astype(float)
    return data
    

class MarkovDecisionProcess:

    def __init__(self, states, actions, transition_model, reward, accessible_states):
        self.states = states # a list
        self.actions = actions # a list
        self.transition_model = transition_model # a callable
        self.reward = reward # a callable
        self.accessible_states = accessible_states # a callable
        self.available_actions = None # a callable
        self.initial_states = None # a list, just for Q-learning
        
    def _sample_state_layout(self):
        s = random.choice(self.states)
        #print(f'Randomly sampled state: {s}')
        available_actions = self.available_actions(s)
        #print(f'Available actions: {available_actions}')
        a = random.choice(available_actions)
        #print(f'Available states from s={s} and a={a}:')
        probs = []
        for s_prime in self.accessible_states(s, a):
            trans_prob = self.transition_model(s,a,s_prime)
            probs.append(trans_prob)
            #print(f'\ts\'={s_prime}; T(s, a, s\')={trans_prob:.3f}; R(s\')={self.reward(s_prime):.3f}')
        #print(f'Total sum of probabilities: {sum(probs):.3f}')

    def policy_iteration(self, gamma=0.99, epsilon=0.01, random_state=0, max_allowed_time=60, max_iter=100):
        '''Solve MDP with policy iteration.
        
        https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node20.html
        '''
        stats = initialize_statistics()
        
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
            iteration_start = time.time()
            if time.time() - start > max_allowed_time:
                #print('Time ran out!')
                break
            if iteration > max_iter:
                #print('Too many iterations.')
                break
            # policy evaluation
            max_change_in_value = epsilon + 1 # anything strictly greater than epsilon
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
                        #print(s, recommended_action)
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
            #print(f'At iteration {iteration}, changed policy values for {changed_values} states.')
            stats = update_stats(stats, iteration, time.time() - iteration_start, changed_values)
                
            if policy_stable:
                terminate_algorithm = True
                
        return pi, V, stats

    def value_iteration(self, gamma=0.99, epsilon=0.01, random_state=0, max_allowed_time=60, max_iter=100):
        stats = initialize_statistics()
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
            iteration_start = time.time()
            if time.time() - start > max_allowed_time:
                #print('Time ran out!')
                break
            if iteration > max_iter:
                #print('Too many iterations.')
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
            #print(f'At iteration {iteration}, max change in value: {max_change_in_value:.5f}; avg. change: {avg_value_change:.5f}')
            
            stats = update_stats(stats, iteration, time.time() - iteration_start, max_change_in_value)
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
                
        return pi, V, stats
        
    def Q_learning(
            self, gamma=0.99, epsilon=0.01, decay_pattern='mitchell', initialization='zeros', exploration='q-optimal',
            random_state=0, max_allowed_time=60, max_iter=10000, iteration_based_decay_factor=0.99, terminate_with='average'
        ):
        # need episodic learning
        # need different decay patterns to use Q_n = alpha_n Q_n-1 + (1 - alpha_n) *(r + gamma Qn-1(s', a'))
        stats = initialize_statistics()
        start = time.time()
        if initialization == 'zeros':
            Q = {
                (s, a): 0
                for s in self.states
                for a in self.available_actions(s)
            }
        elif initialization == 'first_reward':
            rewards = {s: self.reward(s) for s in self.states}
            Q = {
                (s, a): rewards[s]
                for s in self.states
                for a in self.available_actions(s)
            }
        else:
            raise ValueError(f'Unexpected initialization strategy = \'{initialization}\'')
        pi = {
            s: random.choice(self.available_actions(s))
            for s in self.states
        }
        visits = {
            (s, a): 0
            for s in self.states
            for a in self.available_actions(s)
        }
        
        iteration = 0
        
        terminate_algorithm = False
        max_change_in_value = 0
        while not terminate_algorithm:
            max_change_in_value = 0
            change_in_value_seq = []
            iteration += 1
            iteration_start = time.time()
            #print(f'Beginning iteration i={iteration}')
            if time.time() - start > max_allowed_time:
                #print('Time ran out!')
                break
            if iteration > max_iter:
                #print('Too many iterations.')
                break
                
            Q_start = deepcopy(Q)
                
            for s0 in self.initial_states:
                # start at each of the beginning locations
                s = deepcopy(s0)
                first_iteration = True
                # episode
                j = 0
                while s is not None:
                    j += 1
                    if j > 20: break
                    # randomly select an action
                    if exploration == 'introduce-randomness':
                        if random.random() < 0.5:
                            a = random.choice(self.available_actions(s))
                        else:
                            Q_a = {
                                a: Q[(s, a)]
                                for a in self.available_actions(s)
                            }
                            a = max(Q_a, key=Q_a.get)
                    elif exploration == 'q-optimal':
                        Q_a = {
                            a: Q[(s, a)]
                            for a in self.available_actions(s)
                        }
                        a = max(Q_a, key=Q_a.get)
                    else:
                        raise ValueError(f'Unexpected exploration strategy = \'{exploration}\'')
                        
                    if first_iteration:
                        first_iteration = False
                        a0 = deepcopy(a)
                    
                    visits[(s, a)] += 1
                    #print(s, a)
                    
                    if decay_pattern == 'mitchell':
                        # use the approach outlined in the book
                        alpha = 1 / (1 + visits[(s, a)])
                    elif decay_pattern == 'iteration_based':
                        alpha = iteration_based_decay_factor ** iteration
                    else:
                        raise ValueError(f'Unexpected decay_pattern = \'{decay_pattern}\'')
                        
                    # choose s'
                    possible_s_prime = self.accessible_states(s, a)
                    probabilities = [
                        self.transition_model(s, a, s_prime)
                        for s_prime in possible_s_prime
                    ]
                    if len(possible_s_prime) == 0:
                        proposed_value = self.reward(s)
                    else:
                        s_prime = random.choices(possible_s_prime, probabilities)[0]
                        proposed_value = self.reward(s) + gamma * max(
                            Q[(s_prime, a_prime)]
                            for a_prime in self.available_actions(s_prime)
                        )
                    
                    Q[(s, a)] = alpha * Q[(s, a)] + (1 - alpha) * proposed_value
                    
                    if len(possible_s_prime) == 0:
                        s = None
                    else:
                        s = s_prime
                    
                # episode over
                
                beginning_Q_value = Q_start[(s0, a0)]
                ending_Q_value = Q[(s0, a0)]
                max_change_in_value = max(
                    max_change_in_value,
                    abs(ending_Q_value - beginning_Q_value)
                )
                change_in_value_seq.append(abs(ending_Q_value - beginning_Q_value))
                    
            avg_change_in_value = np.average(change_in_value_seq)
            #print(f'At iteration {iteration}, max change in value: {max_change_in_value:.5f}; avg_change_in_value={avg_change_in_value:.5f}')
            stats = update_stats(stats, iteration, time.time() - iteration_start, max_change_in_value)
            if terminate_with == 'max' and max_change_in_value < epsilon:
                terminate_algorithm = True
            elif terminate_with == 'average' and avg_change_in_value < epsilon:
                terminate_algorithm = True
            else:
                assert max_change_in_value > epsilon
                
        for s in pi:
            Q_a = {
                a: Q[(s, a)]
                for a in self.available_actions(s)
            }
            pi[s] = max(Q_a, key=Q_a.get)
            
        return pi, Q, stats
        
    def brute_Q_learning(self, gamma, epsilon=0.001, decay_pattern='mitchell', random_state=0, max_allowed_time=60, max_iter=100):
        start = time.time()
        Q = {
            (s, a): 0
            for s in self.states
            for a in self.available_actions(s)
        }
        pi = {
            s: random.choice(self.available_actions(s))
            for s in self.states
        }
        
        iteration = 0
        
        terminate_algorithm = False
        max_change_in_value = 0
        while not terminate_algorithm:
            max_change_in_value = 0
            iteration += 1
            #print(f'Beginning iteration i={iteration}')
            if time.time() - start > max_allowed_time:
                #print('Time ran out!')
                break
            if iteration > max_iter:
                #print('Too many iterations.')
                break
            for j, (s, a) in enumerate(Q):
                #if j % 100 == 0: print(f'\tj={j}')
                # sample new state s'
                possible_s_prime = self.accessible_states(s, a)
                probabilities = [
                    self.transition_model(s, a, s_prime)
                    for s_prime in possible_s_prime
                ]
                if len(possible_s_prime) == 0:
                    proposed_value = self.reward(s)
                else:
                    s_prime = random.choices(possible_s_prime, probabilities)[0]
                    proposed_value = self.reward(s) + gamma * max(
                        Q[(s_prime, a_prime)]
                        for a_prime in self.available_actions(s_prime)
                    )
                max_change_in_value = max(
                    max_change_in_value,
                    abs(Q[(s, a)] - proposed_value)
                )
                
                if decay_pattern == 'mitchell':
                    # use the approach outlined in the book
                    alpha = 1 / (1 + visits[(s, a)])
                elif decay_pattern == 'iteration_based':
                    alpha = iteration_based_decay_factor ** iteration
                else:
                    raise ValueError(f'Unexpected decay_pattern = \'{decay_pattern}\'')
                    
                Q[(s, a)] = alpha * Q[(s, a)] + (1 - alpha) * proposed_value
            print(f'At iteration {iteration}, max change in value: {max_change_in_value:.5f}')
            if max_change_in_value < epsilon:
                terminate_algorithm = True
                
        for s in pi:
            Q_a = {
                a: Q[(s, a)]
                for a in self.available_actions(s)
            }
            pi[s] = max(Q_a, key=Q_a.get)
            
        return pi, Q
                
        
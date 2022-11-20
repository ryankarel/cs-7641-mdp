"""Set up blackjack problem."""

import numpy as np
from itertools import product
import random

LOSE = -1
WIN = 1

verbose = True


def deal():
    probabilities = np.array([1.0] * 9 + [4.0])
    probabilities /= probabilities.sum()
    random_card_value = np.random.choice(
        range(1, 11),
        p=probabilities
    )
    return random_card_value


def hit(player_value, player_aces, dealer_card):
    return np.random.choice([True, False])


def dealer_hit(cards):
    if verbose: print(f'Dealer cards: {cards}')
    if aces(cards) == 0:
        return sum(cards) < 17
    else:
        return sum(cards) <= 17


def aces(cards):
    num_aces = len([card for card in cards if card == 1])
    return num_aces


def bust(cards):
    return sum(cards) > 21


def hand_value(cards):
    face_value = sum(cards)
    num_aces = aces(cards)
    if num_aces == 0:
        return face_value
    else:
        non_ace_cards = [card for card in cards if card != 1]
        ace_cards = [card for card in cards if card == 1]
        one_11 = sum(non_ace_cards) + sum(ace_cards[:-1]) + 11
        if one_11 <= 21:
            return one_11
        else:
            all_1s = sum(non_ace_cards) + sum(ace_cards)
            return all_1s



def compare_hands(player_cards, dealer_cards):
    if verbose:
        print(f'Player has {player_cards}, dealer has {dealer_cards}')
    if hand_value(player_cards) > hand_value(dealer_cards):
        if verbose: print('Player wins')
        return WIN
    else:
        if verbose: print('Player loses')
        return LOSE


def game():

    player_cards = [deal(), deal()]
    dealer_cards = [deal(), deal()]

    dealer_card = dealer_cards[0]

    while hit(sum(player_cards), aces(player_cards), dealer_card):
        if verbose: print('Player hitting')
        player_cards.append(deal())
        if bust(player_cards):
            if verbose: print('Player bust')
            return LOSE
    if verbose: print('Player stays')

    while dealer_hit(dealer_cards):
        if verbose: print('Dealer hitting')
        dealer_cards.append(deal())
        if bust(dealer_cards):
            if verbose: print('Dealer bust')
            return WIN

    return compare_hands(player_cards, dealer_cards)

game()


class MarkovDecisionProcess:

    def __init__(self, states, actions, transition_model, reward, accessible_states):
        self.states = states # a list
        self.actions = actions # a list
        self.transition_model = transition_model # a callable
        self.reward = reward # a callable
        self.accessible_states = accessible_states # a callable

    def policy_iteration(self, theta, random_state, max_allowed_time, max_iter):
        # randomly initialize value function
        V = dict(zip(
            self.states,
            np.random.random_sample(len(self.states))
        ))

        estimated_policy = dict(zip(
            self.states,
            np.random.random_sample(len(self.states))
        ))

        change_in_value = 0




class BlackjackMDP(MarkovDecisionProcess):

    def __init__(self):
        # here a state is a 4-tuple that captures the total point value
        # of a dealt hand, whether at least one ace is present
        # in this hand, the face up card for the dealer, and finally
        # whether the player is holding or not.
        self.states = list(product(
            list(range(2, 33, 1)),
            [0, 1],
            list(range(1, 11)),
            [0, 1]
        ))
        # remove impossible states
        for dealer_card in range(1, 11):
            self.states.remove((2, 0, dealer_card))
            self.states.remove((3, 0, dealer_card))

        self.actions = [False, True] # to hit or not hit

    def accessible_states(self, s):
        possibilities = []
        if s[-1] == 1:
            return possibilities # [], a terminal state with no remaining options
        # could hold
        holding = s.copy()
        holding[-1] = 1
        possibilities.append(holding)
        if s[0] > 21: # already bust
            return possibilities
        if s[1] == 1: # already has at least one ace
            for i in range(1, 12):
                possible_hit = s.copy()
                possible_hit[0] += i
                possibilities.append(possible_hit)
        else:
            # no aces yet
            for new_ace_presence in [0, 1]:
                for i in range(1, 12):
                    if i == 1 and new_ace_presence == 1:
                        continue
                    possible_hit = s.copy()
                    possible_hit[0] += i
                    possible_hit[1] = new_ace_presence
                    possibilities.append(possible_hit)
        return possibilities

    def transition_model(self, s, a, s_prime):
        if s_prime[2] != s[2]:
            return 0
        if s == s_prime:
            if a: # hitting
                return 0 # no way can this happen
            else:
                return 1
        diff = s_prime[0] - s[0]
        if (diff < 0) or (diff > 11):
            return 0
        if s_prime[1] and not s[1]: # we're definitely getting an ace
            if diff not in [1, 11]:
                return 0
            else:
                return 1 / 13 # probability of getting an ace
        if diff < 10 or diff == 11:
            return 1 / 13
        elif diff == 10:
            return 4 / 13
        else:
            Exception('Didnt expect this case.')

    def reward(self, s):
        if s[3] == 0: # check if we're holding or not
            # here we are not holding, which means no reward (positive or
            # negative) is coming, yet
            return 0
        player_point_value = s[0]
        if player_point_value > 21:
            return LOSE
        elif player_point_value == 21:
            return WIN

        dealer_cards = [s[2], deal()]
        while dealer_hit(dealer_cards):
            dealer_cards.append(deal())
            if bust(dealer_cards):
                return WIN
        if player_point_value > hand_value(dealer_cards):
            return WIN
        else:
            return LOSE




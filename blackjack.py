"""Set up blackjack problem."""

import numpy as np

LOSE = 0
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
    
    
    
    
"""
Monte Carlo rollout agent for 6 Nimmt! (Take 6!)

For each candidate card in hand, simulate N random completions of the game
(opponents draw uniformly from unknown cards) and pick the card with the
lowest expected cumulative penalty for this player.
"""

from __future__ import annotations

import copy
import random

N_CARDS     = 104
BOARD_SIZE_X = 5
MAX_HAND    = 10


# ---------------------------------------------------------------------------
# Duplicated helpers (kept local to avoid coupling to petting_zoo internals)
# ---------------------------------------------------------------------------

def _card_score(card: int) -> int:
    if card % 55 == 0: return 7
    if card % 11 == 0: return 5
    if card % 10 == 0: return 3
    if card %  5 == 0: return 2
    return 1


def _row_score(row: list) -> int:
    return sum(_card_score(c) for c in row)


def _place_card(card: int, board: list) -> int:
    """Place card on board (mutates), return penalty incurred."""
    best_idx, best_end = -1, -1
    for i, row in enumerate(board):
        last = row[-1]
        if last < card and last > best_end:
            best_end = last
            best_idx = i

    if best_idx != -1:
        if len(board[best_idx]) >= BOARD_SIZE_X:
            score = _row_score(board[best_idx])
            board[best_idx] = [card]
        else:
            score = 0
            board[best_idx].append(card)
    else:
        chosen = min(range(len(board)),
                     key=lambda i: (_row_score(board[i]), len(board[i]), i))
        score = _row_score(board[chosen])
        board[chosen] = [card]
    return score


def _resolve_round(board: list, plays: dict) -> dict:
    """Resolve one round. plays = {player_idx: card}. Returns {player_idx: penalty}."""
    penalties = {idx: 0 for idx in plays}
    for idx, card in sorted(plays.items(), key=lambda x: x[1]):
        penalties[idx] = _place_card(card, board)
    return penalties


def _unknown_cards(hand: list, history: dict) -> list:
    """Cards that are not visible to us (not in hand, board, or past plays)."""
    seen = set(hand)
    for row in history.get("board", []):
        seen.update(row)
    for actions in history.get("history_matrix", []):
        seen.update(actions)
    # board_history entries overlap with history_matrix; skip to avoid double-work
    return [c for c in range(1, N_CARDS + 1) if c not in seen]


# ---------------------------------------------------------------------------
# BestPlayer2: Monte Carlo rollout agent
# ---------------------------------------------------------------------------

class BestPlayer2:
    """
    For each card in hand, roll out N random completions and return
    the card that minimises expected personal penalty over the game.

    Opponents are modelled as drawing uniformly from the pool of cards
    not visible to us; our own future plays are also random (we only
    optimise the *current* card choice, not the full sequence).
    """

    def __init__(self, player_idx: int, n_simulations: int = 100):
        self.player_idx   = player_idx
        self.n_simulations = n_simulations

    def action(self, hand: list, history: dict) -> int:
        if len(hand) == 1:
            return hand[0]

        board         = history.get("board", [])
        current_round = history.get("round", 0)
        n_players     = len(history.get("scores", [0] * 4))
        n_opponents   = n_players - 1

        # Cards remaining in each opponent's hand (same as ours before we play)
        opp_hand_size = MAX_HAND - current_round

        unknown = _unknown_cards(hand, history)

        expected = {}

        for candidate in hand:
            my_remaining = [c for c in hand if c != candidate]
            total_penalty = 0.0

            for _ in range(self.n_simulations):
                # Sample opponent hands from unknown pool
                pool = unknown.copy()
                random.shuffle(pool)

                opp_hands: list[list] = []
                for _ in range(n_opponents):
                    size = min(opp_hand_size, len(pool))
                    opp_hands.append(pool[:size])
                    pool = pool[size:]

                sim_board  = copy.deepcopy(board)
                my_hand    = my_remaining.copy()
                my_penalty = 0

                # ── current round: we play `candidate` ──────────────────────
                plays = {self.player_idx: candidate}
                for opp_i, opp_hand in enumerate(opp_hands):
                    opp_idx = opp_i if opp_i < self.player_idx else opp_i + 1
                    if opp_hand:
                        card = random.choice(opp_hand)
                        plays[opp_idx] = card
                        opp_hands[opp_i] = [c for c in opp_hand if c != card]

                my_penalty += _resolve_round(sim_board, plays).get(self.player_idx, 0)

                # ── future rounds: all players random ───────────────────────
                while my_hand:
                    my_card = random.choice(my_hand)
                    my_hand.remove(my_card)

                    plays = {self.player_idx: my_card}
                    for opp_i, opp_hand in enumerate(opp_hands):
                        opp_idx = opp_i if opp_i < self.player_idx else opp_i + 1
                        if opp_hand:
                            card = random.choice(opp_hand)
                            plays[opp_idx] = card
                            opp_hands[opp_i] = [c for c in opp_hand if c != card]

                    my_penalty += _resolve_round(sim_board, plays).get(self.player_idx, 0)

                total_penalty += my_penalty

            expected[candidate] = total_penalty / self.n_simulations

        return min(hand, key=lambda c: expected[c])

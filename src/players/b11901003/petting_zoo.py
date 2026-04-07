"""
PettingZoo AEC environment for 6 Nimmt! (Take 6!)

Turn order: each round, agents select a card one-by-one (AEC style).
After all agents have selected, the round is resolved simultaneously and
rewards are issued as  -(score_incurred_this_round).

Observation (54 floats) matches PlayerBase._embed_state:
  - Board : 4 rows × 3 features           = 12
  - Hand  : 10 slots × 4 features         = 40
  - Context: [round_progress, score_diff]  =  2

Action: integer in [0, 9] — index into the current hand (padded to 10).
        Invalid indices (>= hand size) are clipped to the last valid card.

Usage
-----
    env = SixNimmtAECEnv(n_players=4, opponent_cls=Baseline1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    obs, _ = env.reset()
    ...

Training with Stable-Baselines3 (single-agent wrapper)
-------------------------------------------------------
    from stable_baselines3 import PPO
    env = SixNimmtSingleAgentEnv(opponent_cls=Baseline1)
    model = PPO("MlpPolicy", env, ...)
    model.learn(total_timesteps=...)
"""

from __future__ import annotations

import copy
import random
from typing import Optional

import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

# ---------------------------------------------------------------------------
# Game constants (must match Engine defaults)
# ---------------------------------------------------------------------------
N_CARDS      = 104
N_ROUNDS     = 10
BOARD_SIZE_X = 5   # max cards per row before forced take
BOARD_SIZE_Y = 4   # number of rows
MAX_HAND     = 10  # cards dealt per player
OBS_SIZE     = BOARD_SIZE_Y * 3 + 4 * MAX_HAND + 2  # 54


# ---------------------------------------------------------------------------
# Standalone helpers (mirror PlayerBase without subclassing)
# ---------------------------------------------------------------------------

def _card_score(card: int) -> int:
    if card % 55 == 0: return 7
    if card % 11 == 0: return 5
    if card % 10 == 0: return 3
    if card %  5 == 0: return 2
    return 1


def _row_score(row: list) -> int:
    return sum(_card_score(c) for c in row)


def _compute_presum(hand, history):
    played = set()
    for actions in history.get("history_matrix", []):
        played.update(actions)
    for row in history.get("board_history", []):
        for r in row:
            played.update(r)
    for r in history.get("board", []):
        played.update(r)
    played.update(hand)
    presum = [0] * (N_CARDS + 1)
    for c in range(1, N_CARDS + 1):
        presum[c] = presum[c - 1] + (0 if c in played else 1)
    return presum


def _embed_board(board, presum) -> list:
    emb = []
    for row in board:
        if not row:
            emb.extend([0.0, 0.0, 0.0])
        else:
            emb.append(presum[row[-1]] / N_CARDS)
            emb.append(len(row) / BOARD_SIZE_X)
            emb.append(_row_score(row) / 27.0)
    return emb


def _embed_hand(hand: list, board: list, presum) -> list:
    emb = []
    for card in hand:
        norm_rank = presum[card] / N_CARDS

        best_row, best_last = None, -1
        for row in board:
            last = row[-1] if row else 0
            if last < card and last > best_last:
                best_last = last
                best_row  = row

        if best_row is None:
            fits_row  = 0.0
            takes_row = 1.0
            cheapest  = min(board, key=lambda r: (_row_score(r), len(r)))
            norm_pen  = _row_score(cheapest) / 27.0
        else:
            fits_row = 1.0
            if len(best_row) >= BOARD_SIZE_X:
                takes_row = 1.0
                norm_pen  = _row_score(best_row) / 27.0
            else:
                takes_row = 0.0
                norm_pen  = 0.0

        emb.extend([norm_rank, fits_row, takes_row, norm_pen])

    # pad remaining slots
    emb += [0.0, 0.0, 0.0, 0.0] * (MAX_HAND - len(hand))
    return emb


def _embed_state(hand, history, player_idx) -> np.ndarray:
    board  = history.get("board", [])
    board  = sorted(board, key=lambda r: r[-1] if r else 0)
    presum = _compute_presum(hand, history)

    board_emb = _embed_board(board, presum)
    hand_emb  = _embed_hand(hand, board, presum)

    round_num = history.get("round", 0)
    scores    = history.get("scores", [])
    if scores and len(scores) > player_idx:
        my_score = scores[player_idx]
        opp      = [s for i, s in enumerate(scores) if i != player_idx]
        avg_opp  = sum(opp) / len(opp) if opp else my_score
        score_diff = (my_score - avg_opp) / 50.0
    else:
        score_diff = 0.0

    round_feat = round_num / N_ROUNDS
    return np.array(board_emb + hand_emb + [round_feat, score_diff], dtype=np.float32)


# ---------------------------------------------------------------------------
# Place a card on the board (mutates board in place, returns score incurred)
# ---------------------------------------------------------------------------

def _place_card(card: int, board: list) -> int:
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


# ---------------------------------------------------------------------------
# AEC Environment
# ---------------------------------------------------------------------------

class SixNimmtAECEnv(AECEnv):
    """
    PettingZoo AEC environment for 6 Nimmt!

    Each round proceeds as an AEC sequence: every agent selects a card in
    turn, then the round is resolved simultaneously (lowest card placed first)
    and per-agent rewards are issued.

    Parameters
    ----------
    n_players : int
        Total number of seats (learning agent + opponents), default 4.
    opponent_cls : class | list[class] | None
        Opponent class(es); must implement action(hand, history).
        A single class is replicated for seats 1..n_players-1.
        Pass None to make all seats learning agents.
    seed : int | None
        RNG seed.
    """

    metadata = {
        "render_modes": [],
        "name": "six_nimmt_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        n_players: int = 4,
        opponent_cls=None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.n_players = n_players
        self.seed_val  = seed

        # Build scripted-opponent registry (player index -> instance)
        if opponent_cls is None:
            self._opponent_fns = {}
        else:
            if not isinstance(opponent_cls, (list, tuple)):
                opponent_cls = [opponent_cls] * (n_players - 1)
            assert len(opponent_cls) == n_players - 1
            self._opponent_fns = {
                i + 1: cls(player_idx=i + 1)
                for i, cls in enumerate(opponent_cls)
            }

        self.possible_agents    = [f"player_{i}" for i in range(n_players)]
        self._agent_name_to_idx = {a: i for i, a in enumerate(self.possible_agents)}

        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        act_space = gym.spaces.Discrete(MAX_HAND)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces      = {a: act_space for a in self.possible_agents}

        self._rng = random.Random(seed)
        self._reset_state()

    # ------------------------------------------------------------------
    # PettingZoo interface
    # ------------------------------------------------------------------

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = random.Random(seed)

        self._reset_state()

        self.agents          = list(self.possible_agents)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards             = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {}    for a in self.agents}

        self._pending_actions: dict = {}   # agent_name -> card chosen this round

        obs = {a: self._observe(a) for a in self.agents}
        return obs, self.infos

    def observe(self, agent):
        return self._observe(agent)

    def step(self, action):
        agent = self.agent_selection
        idx   = self._agent_name_to_idx[agent]

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0

        # Clip action to valid hand range
        hand     = self._hands[idx]
        card_idx = max(0, min(int(action), len(hand) - 1))
        self._pending_actions[agent] = hand[card_idx]

        # Advance turn
        self.agent_selection = self._agent_selector.next()

        # Resolve round once all seats have chosen
        if len(self._pending_actions) == len(self.agents):
            self._resolve_round()

        self._accumulate_rewards()

    # ------------------------------------------------------------------
    # Round resolution
    # ------------------------------------------------------------------

    def _resolve_round(self):
        self._board_history.append([row.copy() for row in self._board])

        # Scripted opponents that haven't acted yet (shouldn't happen in normal
        # AEC flow, but handle gracefully)
        history_state = self._make_history_state()
        for seat_idx, opp in self._opponent_fns.items():
            agent_name = self.possible_agents[seat_idx]
            if agent_name not in self._pending_actions:
                hand   = self._hands[seat_idx]
                chosen = opp.action(hand.copy(), copy.deepcopy(history_state))
                if not isinstance(chosen, int) or chosen not in hand:
                    chosen = hand[0]
                self._pending_actions[agent_name] = chosen

        # Record actions and remove cards from hands
        round_actions = [0] * self.n_players
        for agent_name, card in self._pending_actions.items():
            i = self._agent_name_to_idx[agent_name]
            round_actions[i] = card
            self._hands[i].remove(card)
        self._history_matrix.append(round_actions)

        # Place cards in ascending order
        prev_scores  = list(self._scores)
        sorted_plays = sorted(
            ((card, self._agent_name_to_idx[a]) for a, card in self._pending_actions.items()),
            key=lambda x: x[0],
        )
        for card, pidx in sorted_plays:
            self._scores[pidx] += _place_card(card, self._board)

        self._score_history.append(list(self._scores))
        self._round += 1
        self._pending_actions.clear()

        # Reward = -(my_penalty_this_round - avg_opponent_penalty_this_round)
        for agent_name in self.agents:
            pidx       = self._agent_name_to_idx[agent_name]
            my_delta   = self._scores[pidx] - prev_scores[pidx]
            opp_idx    = [i for i in range(self.n_players) if i != pidx]
            avg_opp    = (
                sum(self._scores[i] - prev_scores[i] for i in opp_idx) / len(opp_idx)
                if opp_idx else 0.0
            )
            self.rewards[agent_name] = float(-(my_delta - avg_opp))

        if self._round >= N_ROUNDS:
            for a in self.agents:
                self.terminations[a] = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._round  = 0
        self._scores = [0] * self.n_players

        deck = list(range(1, N_CARDS + 1))
        self._rng.shuffle(deck)

        self._board: list = []
        for _ in range(BOARD_SIZE_Y):
            self._board.append([deck.pop()])

        self._hands: list = []
        for _ in range(self.n_players):
            hand = sorted(deck.pop() for _ in range(N_ROUNDS))
            self._hands.append(hand)

        self._history_matrix: list = []
        self._board_history:  list = []
        self._score_history:  list = []

    def _make_history_state(self) -> dict:
        return {
            "board":          [row.copy() for row in self._board],
            "scores":         list(self._scores),
            "round":          self._round,
            "history_matrix": [r.copy() for r in self._history_matrix],
            "board_history":  [[r.copy() for r in b] for b in self._board_history],
            "score_history":  [s.copy() for s in self._score_history],
        }

    def _observe(self, agent: str) -> np.ndarray:
        idx     = self._agent_name_to_idx[agent]
        hand    = self._hands[idx]
        history = self._make_history_state()
        return _embed_state(hand, history, idx)

    def _accumulate_rewards(self):
        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]
            self.rewards[a] = 0.0

    def render(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Single-agent Gymnasium wrapper (agent 0 vs scripted opponents)
# ---------------------------------------------------------------------------

class SixNimmtSingleAgentEnv(gym.Env):
    """
    Gymnasium wrapper around SixNimmtAECEnv.

    Only "player_0" is a learning agent; all other seats use opponent_cls.
    Useful for Stable-Baselines3 and other single-agent RL libraries.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_cls, n_players: int = 4, seed=None):
        super().__init__()
        self._env = SixNimmtAECEnv(
            n_players=n_players,
            opponent_cls=opponent_cls,
            seed=seed,
        )
        self.observation_space = self._env.observation_space("player_0")
        self.action_space      = self._env.action_space("player_0")

    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self._env.reset(seed=seed, options=options)
        return obs_dict["player_0"], info_dict.get("player_0", {})

    def step(self, action):
        total_reward = 0.0

        # Step player_0
        self._env.step(action)
        total_reward += self._env._cumulative_rewards.get("player_0", 0.0)
        terminated    = self._env.terminations.get("player_0", False)
        truncated     = self._env.truncations.get("player_0",  False)

        # Auto-step scripted agents until it's player_0's turn or episode ends
        while (
            not terminated
            and not truncated
            and self._env.agent_selection != "player_0"
        ):
            agent   = self._env.agent_selection
            idx     = self._env._agent_name_to_idx[agent]
            opp     = self._env._opponent_fns.get(idx)
            if opp is not None:
                hand    = self._env._hands[idx]
                history = self._env._make_history_state()
                chosen  = opp.action(hand.copy(), copy.deepcopy(history))
                if not isinstance(chosen, int) or chosen not in hand:
                    chosen = hand[0]
                act = hand.index(chosen)
            else:
                act = 0
            self._env.step(act)
            total_reward += self._env._cumulative_rewards.get("player_0", 0.0)
            terminated    = self._env.terminations.get("player_0", False)
            truncated     = self._env.truncations.get("player_0",  False)

        obs  = self._env.observe("player_0")
        info = self._env.infos.get("player_0", {})
        return obs, total_reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Inference wrapper: load SB3 model as a player compatible with the Engine
# ---------------------------------------------------------------------------

class PettingZooPlayer:
    """
    Wraps a trained SB3 model so it can be used in place of any player
    inside the existing Engine / tournament runner.

    Usage in configs/game/example.json:
        ["src.players.b11901003.petting_zoo", "PettingZooPlayer",
         {"model_path": "src/players/b11901003/trained/pz_.../best_model.zip"}]
    """

    def __init__(self, player_idx: int, model_path: str):
        from stable_baselines3 import PPO, A2C
        import os

        self.player_idx = player_idx
        # Try PPO first, fall back to A2C
        try:
            self._model = PPO.load(model_path)
        except Exception:
            self._model = A2C.load(model_path)

        # Build a throwaway env to reuse observation helpers
        self._dummy = SixNimmtAECEnv.__new__(SixNimmtAECEnv)
        self._dummy.n_players   = 4
        self._dummy._rng        = random.Random()
        self._dummy._opponent_fns = {}
        self._dummy._agent_name_to_idx = {f"player_{i}": i for i in range(4)}

    def action(self, hand: list, history: dict) -> int:
        obs = _embed_state(hand, history, self.player_idx)
        act, _ = self._model.predict(obs, deterministic=True)
        act = max(0, min(int(act), len(hand) - 1))
        return hand[act]


# ---------------------------------------------------------------------------
# Training entry point
#   python -m src.players.b11901003.petting_zoo
#   python -m src.players.b11901003.petting_zoo --algo ppo --timesteps 500000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import importlib
    import os
    import sys
    import time

    REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    sys.path.insert(0, os.path.abspath(REPO_ROOT))

    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       type=str, default="ppo",
                        choices=["ppo", "a2c"])
    parser.add_argument("--timesteps",  type=int, default=500_000)
    parser.add_argument("--n_envs",     type=int, default=8)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--opponent",   type=str,
                        default="src.players.TA.public_baselines1:Baseline1")
    parser.add_argument("--output_dir", type=str,
                        default="src/players/b11901003/trained")
    parser.add_argument("--load",       type=str, default=None)
    args = parser.parse_args()

    mod_path, cls_name = args.opponent.rsplit(":", 1)
    OpponentCls = getattr(importlib.import_module(mod_path), cls_name)

    run_tag    = f"pz_{args.algo}_{args.lr}_{args.timesteps}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(os.path.abspath(REPO_ROOT), args.output_dir, run_tag)
    os.makedirs(output_dir, exist_ok=True)

    def make_env():
        return SixNimmtSingleAgentEnv(opponent_cls=OpponentCls)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs)

    AlgoCls = PPO if args.algo == "ppo" else A2C
    if args.load:
        model = AlgoCls.load(args.load, env=vec_env)
        print(f"Resumed from {args.load}")
    else:
        model = AlgoCls(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            verbose=1,
            tensorboard_log=os.path.join(output_dir, "tb"),
        )

    eval_callback = EvalCallback(
        SixNimmtSingleAgentEnv(opponent_cls=OpponentCls),
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=200,
        deterministic=True,
        verbose=1,
    )

    print(f"Training {args.algo.upper()} for {args.timesteps:,} timesteps ...")
    print(f"Output : {output_dir}")

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(os.path.join(output_dir, "final_model"))
    print(f"Done. Model saved to {output_dir}/final_model.zip")

import torch
import torch.nn as nn
import torch.optim as optim
from .model import Model
from .playerBase import PlayerBase


class RLPlayer(PlayerBase):
    """
    Actor-Critic (REINFORCE with learned baseline) agent for 6 Nimmt!

    Improvements over vanilla REINFORCE:
    - Shared backbone with separate policy and value heads (actor-critic).
    - Advantage = discounted return - value baseline (variance reduction).
    - Entropy bonus to maintain exploration throughout training.
    - Relative reward: each round's reward is penalized relative to opponents,
      so the agent learns to beat others, not just minimise absolute score.
    - Gradient clipping for stable updates.

    Training usage:
        agent.model.train()
        engine.play_game()
        loss = agent.update(engine.score_history)

    Inference usage:
        agent.model.eval()
        card = agent.action(hand, history)
    """

    # board has board_size_y=4 rows; _embed_board returns 4×3=12 values
    # hand embedding: 10 values
    # context: round_feat + score_diff = 2 values
    N_ROWS = 4
    INPUT_SIZE = N_ROWS * 3 + 10 + 2   # 24

    def __init__(self, player_idx, lr=1e-3, gamma=0.99, batch_size=32,
                 entropy_coef=0.01, value_coef=0.5, checkpoint=None):
        super().__init__(player_idx)
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.model = Model(input_size=self.INPUT_SIZE, hidden_size=128, output_size=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Per-game buffers
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_entropies = []

        # Batch accumulation buffers
        self._pending_log_probs = []
        self._pending_values = []
        self._pending_entropies = []
        self._pending_returns = []
        self._games_in_batch = 0

        if checkpoint is not None:
            self.load(checkpoint)
            self.model.eval()

    def action(self, hand, history):
        state_emb = self._embed_state(hand, history)
        state_t = torch.tensor(state_emb, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(
            [1.0 if i < len(hand) else 0.0 for i in range(self.max_num_card_in_hand)],
            dtype=torch.float32,
        ).unsqueeze(0)

        if self.model.training:
            logits, value = self.model(state_t, mask_t)
            logits, value = logits.squeeze(0), value.squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            local_idx = dist.sample()
            self.saved_log_probs.append(dist.log_prob(local_idx))
            self.saved_values.append(value)
            self.saved_entropies.append(dist.entropy())
            return hand[local_idx.item()]
        else:
            with torch.no_grad():
                logits, _ = self.model(state_t, mask_t)
                local_idx = logits.squeeze(0).argmax().item()
            return hand[local_idx]

    def update(self, score_history):
        """
        Call once per game. Accumulates into a batch; performs a gradient step
        when batch_size complete games have been collected.

        Args:
            score_history: engine.score_history — list of length n_rounds where
                           each element is cumulative scores of all players.
        Returns:
            loss (float) if a gradient step was taken, else None.
        """
        if not self.saved_log_probs:
            return None

        # --- per-round relative rewards ---
        # reward = -(my_penalty - avg_opponent_penalty)
        # Positive reward when we take fewer bull-heads than the average opponent.
        rewards = []
        prev_scores = None
        for round_scores in score_history:
            n = len(round_scores)
            if prev_scores is None:
                my_penalty = round_scores[self.player_idx]
                opp_total = sum(round_scores) - round_scores[self.player_idx]
                avg_opp_penalty = opp_total / (n - 1) if n > 1 else 0.0
            else:
                my_penalty = round_scores[self.player_idx] - prev_scores[self.player_idx]
                opp_now = sum(round_scores) - round_scores[self.player_idx]
                opp_prev = sum(prev_scores) - prev_scores[self.player_idx]
                avg_opp_penalty = (opp_now - opp_prev) / (n - 1) if n > 1 else 0.0
            rewards.append(-(my_penalty - avg_opp_penalty))
            prev_scores = round_scores

        assert len(rewards) == len(self.saved_log_probs), (
            f"Reward/log-prob length mismatch: {len(rewards)} vs {len(self.saved_log_probs)}"
        )

        # --- discounted returns ---
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # --- accumulate into batch buffers ---
        self._pending_log_probs.extend(self.saved_log_probs)
        self._pending_values.extend(self.saved_values)
        self._pending_entropies.extend(self.saved_entropies)
        self._pending_returns.extend(returns)
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_entropies = []
        self._games_in_batch += 1

        if self._games_in_batch < self.batch_size:
            return None

        # --- gradient step ---
        returns_t = torch.tensor(self._pending_returns, dtype=torch.float32)
        values_t = torch.stack(self._pending_values)
        log_probs_t = torch.stack(self._pending_log_probs)
        entropies_t = torch.stack(self._pending_entropies)

        # Normalize returns for variance reduction
        if returns_t.numel() > 1:
            std = returns_t.std(correction=0)
            if std > 1e-8:
                returns_t = (returns_t - returns_t.mean()) / (std + 1e-8)

        advantages = returns_t - values_t.detach()

        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = nn.functional.mse_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # reset batch state
        self._pending_log_probs = []
        self._pending_values = []
        self._pending_entropies = []
        self._pending_returns = []
        self._games_in_batch = 0

        return loss.item()

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


# ---------------------------------------------------------------------------
# Training entry point
#   python -m src.players.b11901003.rlPlayer
#   python -m src.players.b11901003.rlPlayer --config configs/train/example.json
#   python -m src.players.b11901003.rlPlayer --config configs/train/example.json --load src/players/b11901003/trained/rl_checkpoint.pt
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import sys
    import os
    import json
    import shutil
    import argparse
    import copy
    import importlib

    REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    sys.path.insert(0, os.path.abspath(REPO_ROOT))

    from src.engine import Engine

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/train/example.json",
        help="path to training config JSON (relative to repo root)",
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="resume from an existing checkpoint (.pt file)",
    )
    args = parser.parse_args()

    config_path = os.path.join(os.path.abspath(REPO_ROOT), args.config)
    with open(config_path) as f:
        cfg = json.load(f)

    engine_cfg = cfg["engine"]
    train_cfg  = cfg["train"]
    output_dir = os.path.join(
        os.path.abspath(REPO_ROOT),
        cfg["output_dir"] + f"/rl_{train_cfg['lr']}_{train_cfg['gamma']}_"
                            f"{train_cfg['episodes']}_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(output_dir, os.path.basename(config_path)))

    ckpt_path = os.path.join(output_dir, "rl_checkpoint.pt")

    def load_opponent_cls(spec):
        mod = importlib.import_module(spec[0])
        return getattr(mod, spec[1])

    opponent_specs = cfg["opponents"]
    n_opponents    = len(opponent_specs)
    assert n_opponents == engine_cfg["n_players"] - 1

    agent = RLPlayer(
        player_idx=0,
        lr=train_cfg["lr"],
        gamma=train_cfg["gamma"],
        batch_size=train_cfg.get("batch_size", 1),
    )
    if args.load:
        agent.load(args.load)
        print(f"Resumed from {args.load}")

    agent.model.train()

    episodes   = train_cfg["episodes"]
    log_every  = train_cfg["log_every"]
    batch_size = train_cfg.get("batch_size", 1)

    total_loss  = 0.0
    n_updates   = 0
    total_score = 0.0
    best_avg    = float("inf")

    print(f"Training for {episodes} episodes (batch_size={batch_size}) …")
    print(f"Config : {args.config}")
    print(f"Output : {output_dir}")

    for ep in range(1, episodes + 1):
        opponents = [
            load_opponent_cls(spec)(player_idx=i + 1)
            for i, spec in enumerate(opponent_specs)
        ]
        players = [agent] + opponents

        engine = Engine(copy.deepcopy(engine_cfg), players)
        final_scores, _ = engine.play_game()

        loss = agent.update(engine.score_history)

        if loss is not None:
            total_loss += loss
            n_updates  += 1
        total_score += final_scores[0]

        if ep % log_every == 0:
            avg_loss  = total_loss / n_updates if n_updates else 0.0
            avg_score = total_score / log_every
            print(f"ep {ep:>6}  avg_loss={avg_loss:+.4f}  avg_penalty={avg_score:.2f}"
                  f"  updates={n_updates}")
            total_loss  = 0.0
            n_updates   = 0
            total_score = 0.0

            if avg_score < best_avg:
                best_avg = avg_score
                agent.save(ckpt_path)
                print(f"          => saved {ckpt_path}  (best avg_penalty={best_avg:.2f})")

    agent.save(ckpt_path)
    print(f"Training complete. Checkpoint: {ckpt_path}")

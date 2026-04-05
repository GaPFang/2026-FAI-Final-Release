import torch
import torch.optim as optim
from .model import Model
from .playerBase import PlayerBase


class RLPlayer(PlayerBase):
    """
    Policy-gradient (REINFORCE) agent for 6 Nimmt!

    Training usage:
        agent.model.train()
        scores, history = engine.play_game()
        loss = agent.update(history["score_history"])   # score_history added in engine

    Inference usage:
        agent.model.eval()
        card = agent.action(hand, history)
    """

    # board has board_size_y=4 rows; _embed_board returns 4 * 3 = 12 values
    N_ROWS = 4
    INPUT_SIZE = 104 * 3 + 104 + 104  # board embedding + hand binary + remaining cards binary

    def __init__(self, player_idx, lr=1e-3, gamma=0.99, batch_size=32, checkpoint=None):
        super().__init__(player_idx)
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = Model(input_size=self.INPUT_SIZE, hidden_size=128, output_size=self.n_cards)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.saved_log_probs = []   # log-probs for the current in-progress game
        self._pending_log_probs = []  # accumulated across games in the current batch
        self._pending_returns = []    # corresponding discounted returns
        self._games_in_batch = 0

        if checkpoint is not None:
            self.load(checkpoint)
            self.model.eval()

    def action(self, hand, history):
        state_emb = self._embed_state(hand, history)
        state_t = torch.tensor(state_emb, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(self._embed_hand(hand), dtype=torch.float32).unsqueeze(0)

        # Indices of cards in hand into the 0-indexed card space
        hand_indices = torch.tensor([c - 1 for c in hand], dtype=torch.long)

        if self.model.training:
            logits = self.model(state_t, mask_t).squeeze(0)        # (n_cards,)
            hand_logits = logits[hand_indices]                      # (|hand|,)
            dist = torch.distributions.Categorical(logits=hand_logits)
            local_idx = dist.sample()                               # index into hand list
            self.saved_log_probs.append(dist.log_prob(local_idx))
            return hand[local_idx.item()]
        else:
            with torch.no_grad():
                logits = self.model(state_t, mask_t).squeeze(0)
                local_idx = logits[hand_indices].argmax().item()
                zipped = list(zip(hand, logits[hand_indices].tolist()))
                print(f"Player {self.player_idx} hand logits: {zipped}")
            return hand[local_idx]

    def update(self, score_history):
        """
        Call once per game. Accumulates the trajectory into a batch; performs a
        gradient update only when batch_size complete games have been collected.

        Args:
            score_history: engine.score_history — list of length n_rounds where
                           each element is the cumulative scores of all players.
        Returns:
            loss (float) if a gradient step was taken this call, else None.
        """
        if not self.saved_log_probs:
            return None

        # --- compute per-round rewards for this game ---
        rewards, prev = [], 0
        for round_scores in score_history:
            penalty = round_scores[self.player_idx] - prev
            rewards.append(-float(penalty))
            prev = round_scores[self.player_idx]

        assert len(rewards) == len(self.saved_log_probs), (
            f"Reward/log-prob length mismatch: {len(rewards)} vs {len(self.saved_log_probs)}"
        )

        # --- discounted returns for this game ---
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # --- accumulate into batch buffers ---
        self._pending_log_probs.extend(self.saved_log_probs)
        self._pending_returns.extend(returns)
        self.saved_log_probs = []
        self._games_in_batch += 1

        if self._games_in_batch < self.batch_size:
            return None  # batch not full yet

        # --- batch is full: gradient step ---
        returns_t = torch.tensor(self._pending_returns, dtype=torch.float32)

        # normalize across the whole batch for variance reduction
        if returns_t.numel() > 1 and returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -torch.stack(
            [lp * G for lp, G in zip(self._pending_log_probs, returns_t)]
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # reset batch state
        self._pending_log_probs = []
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
# Training entry point — reads configs/train/example.json by default
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

    # repo root on path when run as a module from any cwd
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

    # ---- load config -------------------------------------------------------
    config_path = os.path.join(os.path.abspath(REPO_ROOT), args.config)
    with open(config_path) as f:
        cfg = json.load(f)

    engine_cfg  = cfg["engine"]
    train_cfg   = cfg["train"]
    output_dir  = os.path.join(os.path.abspath(REPO_ROOT), cfg["output_dir"] + f"/rl_{train_cfg['lr']}_{train_cfg['gamma']}_{train_cfg['episodes']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    # copy config into output dir so the run is self-contained
    shutil.copy(config_path, os.path.join(output_dir, os.path.basename(config_path)))

    ckpt_path = os.path.join(output_dir, "rl_checkpoint.pt")

    # ---- build opponent constructors from config ---------------------------
    def load_opponent_cls(spec):
        module_path, class_name = spec[0], spec[1]
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)

    opponent_specs = cfg["opponents"]
    n_opponents    = len(opponent_specs)
    assert n_opponents == engine_cfg["n_players"] - 1, (
        f"opponents list has {n_opponents} entries but engine expects "
        f"{engine_cfg['n_players'] - 1}"
    )

    # ---- init agent --------------------------------------------------------
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

    # ---- training loop -----------------------------------------------------
    episodes   = train_cfg["episodes"]
    log_every  = train_cfg["log_every"]
    batch_size = train_cfg.get("batch_size", 1)

    total_loss   = 0.0
    n_updates    = 0       # gradient steps taken in this log window
    total_score  = 0.0
    best_avg     = float("inf")

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

        loss = agent.update(engine.score_history)  # None until batch is full

        if loss is not None:
            total_loss += loss
            n_updates  += 1
        total_score += final_scores[0]   # agent penalty (lower = better)

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

    # always save final weights
    agent.save(ckpt_path)
    print(f"Training complete. Checkpoint: {ckpt_path}")


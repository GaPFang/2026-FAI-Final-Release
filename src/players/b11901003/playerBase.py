class PlayerBase():
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.n_cards = 104
        self.n_rounds = 10
        self.board_size_x = 5  # max cards per row before taking
        self.max_num_card_in_hand = 10
        self.presum = [0] * (self.n_cards + 1)

    def _get_card_score(self, card):
        if card % 55 == 0: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def _get_row_score(self, row):
        return sum(self._get_card_score(c) for c in row)

    def _embed_board(self, board):
        """
        Convert board to a fixed-size embedding.
        Each row: normalized rank of last card, normalized length, normalized score.
        """
        embedding = []
        for row in board:
            if not row:
                embedding.extend([0.0, 0.0, 0.0])
                continue
            norm_last_card = self.presum[row[-1]] / self.n_cards
            norm_length = len(row) / self.board_size_x
            norm_score = self._get_row_score(row) / 27.0  # max row score = 7+5+5+5+5=27
            embedding.extend([norm_last_card, norm_length, norm_score])
        return embedding

    def _embed_hand(self, hand):
        """
        Normalized rank (among remaining cards) for each hand slot; pads to max size.
        Hand is always sorted ascending by the engine, so slot 0 = smallest card.
        """
        embedding = [self.presum[c] / self.n_cards for c in hand]
        embedding += [0.0] * (self.max_num_card_in_hand - len(hand))
        return embedding

    def _compute_presum(self, hand, history):
        played = set()
        for actions in history.get("history_matrix", []):
            played.update(actions)
        for row in history.get("board_history", []):
            for r in row:
                played.update(r)
        for r in history.get("board", []):
            played.update(r)
        self.presum[0] = 0
        for c in range(1, self.n_cards + 1):
            self.presum[c] = self.presum[c-1] + (0 if c in played else 1)

    def _get_remaining_cards(self, hand, history):
        played = set()
        for actions in history.get("history_matrix", []):
            played.update(actions)
        for row in history.get("board_history", []):
            for r in row:
                played.update(r)
        for r in history.get("board", []):
            played.update(r)
        return [0 if c in played else 1 for c in range(1, self.n_cards + 1)]

    def _embed_state(self, hand, history):
        board = history.get("board", [])
        board = sorted(board, key=lambda r: r[-1] if r else 0)
        self._compute_presum(hand, history)
        board_emb = self._embed_board(board)
        hand_emb = self._embed_hand(hand)

        # Game-progress and relative-score context
        round_num = history.get("round", 0)
        scores = history.get("scores", [])
        if scores and len(scores) > self.player_idx:
            my_score = scores[self.player_idx]
            opp = [s for i, s in enumerate(scores) if i != self.player_idx]
            avg_opp = sum(opp) / len(opp) if opp else my_score
            # Positive = we're losing relative to opponents; negative = winning.
            # Divide by 50 as a soft normalizer (typical mid-game differential).
            score_diff = (my_score - avg_opp) / 50.0
        else:
            score_diff = 0.0
        round_feat = round_num / self.n_rounds

        return board_emb + hand_emb + [round_feat, score_diff]

    def action(self, hand, history):
        raise NotImplementedError("Subclasses should implement this!")

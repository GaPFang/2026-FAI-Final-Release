class PlayerBase():
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.n_cards = 104
        self.board_size_x = 5  # max cards per row before taking
        self.max_num_card_in_hand = 10
        self.num_card_in_hand = 10
        self.presum = [0] * (self.n_cards + 1)
    
    def _get_card_score(self, card):
        if card % 55 == 0: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def _get_row_score(self, row):
        return sum(self._get_card_score(c) for c in row)
    
    def _embed_board(self, board, hand):
        """
        Convert board and hand to a fixed-size embedding.
        Each row is represented by its normalized last card, length, and score.
        """
        embedding = []
        for row in board:
            norm_last_card = self.presum[row[-1]] / self.n_cards
            norm_length = len(row) / 5.0
            norm_score = self._get_row_score(row) / 25.0  # 25 is a safe upper bound for row penalties
            embedding.extend([norm_last_card, norm_length, norm_score])
        return embedding

    def _embed_hand(self, hand):
        """
        Convert hand to a fixed-size embedding.
        We use a binary vector indicating which cards are in hand.
        """
        embedding = []
        for hand_card in hand:
            norm_hand_card = self.presum[hand_card] / self.n_cards
            embedding.append(norm_hand_card)
        self.num_card_in_hand = len(hand)
        for _ in range(self.max_num_card_in_hand - self.num_card_in_hand):
            embedding.append(0)  # padding for hands with fewer than max cards
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

    # def _embed_hand(self, hand):
    #     """
    #     Convert hand to a fixed-size embedding.
    #     We use a binary vector indicating which cards are in hand.
    #     """
    #     hand_set = set(hand)
    #     return [1 if c in hand_set else 0 for c in range(1, self.n_cards + 1)]
    
    def _embed_state(self, hand, history):
        board = history.get("board", [])
        board = sorted(board, key=lambda r: r[-1] if r else 0)  # sort by last card for consistency
        self._compute_presum(hand, history)
        board_emb = self._embed_board(board, hand)
        hand_emb = self._embed_hand(hand)
        return board_emb + hand_emb

    def action(self, hand, history):
        raise NotImplementedError("Subclasses should implement this!")


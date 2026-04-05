class PlayerBase():
    def __init__(self, player_idx):
        self.player_idx = player_idx
        self.n_cards = 104
        self.board_size_x = 5  # max cards per row before taking
    
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
        Each row is represented by its last card and length.
        """
        embedding = []
        for row in board:
            embedding.append([row[-1], len(row), self._get_row_score(row)])
        embedding = sorted(embedding, key=lambda x: x[0])
        embedding = sorted(embedding, key=lambda x: x[1])
        embedding = sorted(embedding, key=lambda x: x[2])
        embedding = [item for sublist in embedding for item in sublist]
        return embedding

    def _get_remaining_cards(self, hand, history):
        played = set()
        for actions in history.get("history_matrix", []):
            played.update(actions)
        for row in history.get("board_history", []):
            for r in row:
                played.update(r)
        return [0 if c in played else 1 for c in range(1, self.n_cards + 1)]

    def _embed_hand(self, hand):
        """
        Convert hand to a fixed-size embedding.
        We use a binary vector indicating which cards are in hand.
        """
        hand_set = set(hand)
        return [1 if c in hand_set else 0 for c in range(1, self.n_cards + 1)]
    
    def _embed_state(self, hand, history):
        board_emb = self._embed_board(history["board"])
        hand_emb = self._embed_hand(hand)
        remaining_emb = self._get_remaining_cards(hand, history)
        return board_emb + hand_emb + remaining_emb
    
    def action(self, hand, history):
        raise NotImplementedError("Subclasses should implement this!")


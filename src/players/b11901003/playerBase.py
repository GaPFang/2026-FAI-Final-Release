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
        Creates a (104 x 3) flattened array.
        For every card from 1 to 104, it stores [is_active, normalized_length, normalized_score].
        """
        embedding = []
        
        # Create a quick lookup dictionary for the active rows
        # Key: last_card, Value: (norm_length, norm_score)
        row_stats = {
            row[-1]: (len(row) / 5.0, self._get_row_score(row) / 25.0) 
            for row in board
        }
        
        # Build the 104 slots
        for card in range(1, self.n_cards + 1):
            if card in row_stats:
                length_norm, score_norm = row_stats[card]
                embedding.extend([1.0, length_norm, score_norm])
            else:
                # Empty slot for cards not currently acting as row ends
                embedding.extend([0.0, 0.0, 0.0])
                
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


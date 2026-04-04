class Player1():
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

    def _simulate_placement(self, card, board):
        """
        Simulate placing `card` on `board`.
        Returns (penalty_incurred, new_board).
        Does NOT modify the original board.
        """
        board = [row[:] for row in board]

        best_row = -1
        best_end = -1
        for i, row in enumerate(board):
            end = row[-1]
            if end < card and end > best_end:
                best_end = end
                best_row = i

        penalty = 0
        if best_row != -1:
            if len(board[best_row]) >= self.board_size_x:
                penalty = self._get_row_score(board[best_row])
                board[best_row] = [card]
            else:
                board[best_row].append(card)
        else:
            # Low card: must take the cheapest row
            chosen = min(range(len(board)),
                         key=lambda i: (self._get_row_score(board[i]), len(board[i]), i))
            penalty = self._get_row_score(board[chosen])
            board[chosen] = [card]

        return penalty, board

    def _danger_score(self, board, remaining_cards):
        """
        Estimate danger of the current board state.
        A row with 4 cards is dangerous: any card that fits after it but before
        the next row will trigger a take. More such remaining cards = more danger.
        """
        row_ends = sorted(row[-1] for row in board)
        danger = 0

        for row in board:
            if len(row) < self.board_size_x - 1:
                continue
            end = row[-1]
            row_bh = self._get_row_score(row)

            # Find upper bound: the smallest row end larger than this one
            upper = self.n_cards + 1
            for e in row_ends:
                if e > end:
                    upper = e
                    break

            # Count remaining unknown cards that would land on this row
            trigger_count = sum(1 for c in remaining_cards if end < c < upper)

            # Weight danger by bullheads and how many cards could trigger it
            danger += row_bh * trigger_count * 0.1

        return danger

    def _get_remaining_cards(self, hand, history):
        """Cards not accounted for: not in hand, not played, not currently on board."""
        accounted = set(hand)
        for row in history["board"]:
            accounted.update(row)
        for round_actions in history.get("history_matrix", []):
            for c in round_actions:
                if c > 0:
                    accounted.add(c)
        return [c for c in range(1, self.n_cards + 1) if c not in accounted]

    def action(self, hand, history):
        board = history["board"]
        remaining = self._get_remaining_cards(hand, history)
        remaining_set = set(remaining)

        best_card = hand[0]
        best_cost = float('inf')

        for card in hand:
            penalty, new_board = self._simulate_placement(card, board)

            # Remaining cards after playing this one (opponents still have theirs)
            danger = self._danger_score(new_board, remaining_set)

            # Small bonus for avoiding "low card" forced takes when penalty is 0
            # (we already paid 0, but danger is the main concern)
            total_cost = penalty + danger

            if total_cost < best_cost:
                best_cost = total_cost
                best_card = card

        return best_card

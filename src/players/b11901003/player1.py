class Player1():
    def __init__(self, player_idx):
        self.player_idx = player_idx
    
    def _get_card_score(self, card):
        if card % 55 == 0: return 7
        if card % 11 == 0: return 5
        if card % 10 == 0: return 3
        if card % 5 == 0: return 2
        return 1

    def _print_board(self, board):
        for i, row in enumerate(board):
            row_score = sum(self._get_card_score(c) for c in row)
            print(f"Row {i}: {row} ({row_score} pts)")

    def _print_scores(self, scores):
        print(", ".join(f"P{i}: {score}" for i, score in enumerate(scores)))

    def action(self, hand, history):
        if history["round"] > 0 and history.get("history_matrix") and history.get("board_history") and history.get("score_history"):
            last_actions = history["history_matrix"][-1]
            board_before = history["board_history"][-1]
            score_after = history["score_history"][-1]
            board_after = history["board"]
            if len(history["score_history"]) >= 2:
                score_before = history["score_history"][-2]
            else:
                score_before = [0] * len(score_after)

            print("\nLast round actions")
            print(", ".join(f"P{i}: {card}" for i, card in enumerate(last_actions)))

            print("\nBoard after last round")
            self._print_board(board_after)

            print("\nScore after last round")
            self._print_scores(score_after)
        else:
            print("Current Board:")
            self._print_board(history["board"])
        
        print("\nYour Hand:", hand)
        
        

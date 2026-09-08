"""Termination policies shared by reusable problem evaluators."""


KNIGHT_TOUR_OPEN_SCORE = 63


def knight_tour_complete(best_score: int) -> bool:
    """Return whether all 63 moves of an open 8x8 Knight's Tour exist."""
    return best_score >= KNIGHT_TOUR_OPEN_SCORE

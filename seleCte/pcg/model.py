class MCModel:
    def __init__(self) -> None:
        self.dmatrix = [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 2],
        ]

def fitness_evaluation(candidate):
    """
    Read the name
    """
    # penalizes if no or too many exits (BUT can be holes tho, needs to specify this)
    score_exits = 1000 * get_nb_exits(candidate) * (4 - get_nb_exits(candidate))

    return score_exits

def get_nb_exits(candidate):
    """
    Same
    """
    raise NotImplementedError

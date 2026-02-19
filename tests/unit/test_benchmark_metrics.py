from scripts.benchmark_retrieval import _reciprocal_rank


def test_reciprocal_rank_found() -> None:
    ranked = ["q1", "q2", "q3"]
    assert _reciprocal_rank("q2", ranked) == 0.5


def test_reciprocal_rank_not_found() -> None:
    ranked = ["q1", "q2", "q3"]
    assert _reciprocal_rank("qx", ranked) == 0.0

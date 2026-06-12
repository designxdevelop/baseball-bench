from baseball_bench.scoring.decisions import score_decision_answer
from baseball_bench.tracks.decisions import load_situations


def test_decision_scorer_detects_optimal_action():
    situation = load_situations()[0]
    value, metadata = score_decision_answer('{"action_id": "steal"}', situation.action_values)

    assert value == 1.0
    assert metadata["best_action"] == "steal"
    assert metadata["near_optimal"] is True


def test_decision_scorer_penalizes_invalid_action():
    situation = load_situations()[0]
    value, metadata = score_decision_answer('{"action_id": "swing_for_the_fences"}', situation.action_values)

    assert value == 0.0
    assert metadata["chosen_wp"] is None


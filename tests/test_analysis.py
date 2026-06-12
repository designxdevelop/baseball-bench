from baseball_bench.data import build_database
from baseball_bench.tracks.analysis.question_bank import generate_questions
from baseball_bench.scoring.analysis import score_analysis_answer


def test_generate_questions_uses_database_answers(tmp_path):
    database_path = tmp_path / "baseball.duckdb"
    build_database(database_path)
    questions = generate_questions(database_path=database_path)

    highest_average = next(question for question in questions if question.id == "highest_batting_average_2025")
    assert highest_average.expected_answer == "Mason Cole"


def test_analysis_scoring_honors_tolerance():
    value, metadata = score_analysis_answer('{"answer": 0.386}', "0.387", tolerance=0.001)

    assert value == 1.0
    assert metadata["observed_answer"] == "0.386"


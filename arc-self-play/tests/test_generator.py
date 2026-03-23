"""Tests for puzzle generation."""

from arc_self_play.generator import PuzzleGenerator, PuzzleValidator, GeneratedPuzzle


def test_generate_easy():
    gen = PuzzleGenerator(seed=1)
    puzzle = gen.generate(difficulty="easy")
    assert puzzle is not None
    assert puzzle.difficulty == "easy"
    assert len(puzzle.train) == 3
    assert len(puzzle.transformation) == 1


def test_generate_hard():
    gen = PuzzleGenerator(seed=2)
    puzzle = gen.generate(difficulty="hard")
    assert puzzle is not None
    assert len(puzzle.transformation) >= 3


def test_generate_batch():
    gen = PuzzleGenerator(seed=3)
    puzzles = gen.generate_batch(20)
    assert len(puzzles) == 20
    difficulties = {p.difficulty for p in puzzles}
    assert len(difficulties) >= 2  # Should have mix


def test_non_trivial():
    gen = PuzzleGenerator(seed=4)
    for _ in range(10):
        puzzle = gen.generate()
        if puzzle is None:
            continue
        for inp, out in puzzle.train:
            assert inp != out, "training pair must be non-trivial"
        assert puzzle.test[0] != puzzle.test[1], "test must be non-trivial"


def test_validator_rejects_identity():
    train = [([[1]], [[1]])]
    test = ([[2]], [[2]])
    ok, reason = PuzzleValidator.validate(train, test)
    assert not ok
    assert "identity" in reason


def test_to_dict_roundtrip():
    gen = PuzzleGenerator(seed=5)
    puzzle = gen.generate()
    assert puzzle is not None
    d = puzzle.to_dict()
    assert d["puzzle_id"] == puzzle.puzzle_id
    assert d["difficulty"] == puzzle.difficulty
    assert len(d["train"]) == len(puzzle.train)

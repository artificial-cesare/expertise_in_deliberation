"""Toy checks for the reproduce/ scripts. Run from the repo root: python tests/test_reproduce.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reproduce"))
from correlations import assembly_correlations  # noqa: E402
from temporal_kw import kruskal_by_window, load_assembly, thirds  # noqa: E402


def write(path: Path, **cols) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(path, index=False)


def test_thirds():
    assert thirds(7) == ["start"] * 2 + ["middle"] * 2 + ["end"] * 3
    assert thirds(3) == ["start", "middle", "end"]


def test_windows_are_per_file_and_set_before_filtering():
    with tempfile.TemporaryDirectory() as tmp:
        write(Path(tmp, "a.csv"), num_non_function_words=[3, 9, 9, 9, 9, 9], x=range(6))
        write(Path(tmp, "b.csv"), num_non_function_words=[9, 9, 9], x=range(3))
        df = load_assembly(Path(tmp), min_words=5)
    # a's first row is dropped, but row 1 keeps its "start" position
    assert df["window"].tolist() == ["start", "middle", "middle", "end", "end", "start", "middle", "end"]


def test_epsilon_squared():
    df = pd.DataFrame({"window": ["start"] * 4 + ["middle"] * 4 + ["end"] * 4, "x": list(range(12))})
    res = kruskal_by_window(df, "x")
    h, p = kruskal(range(4), range(4, 8), range(8, 12))
    assert np.isclose(res["H"], h) and np.isclose(res["p"], p)
    assert np.isclose(res["eps2"], h / (12 - 1))


def test_scotland_llm_drop_leaves_irish_only():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        for folder in ("8th_amendment", "gendereq", "scotland"):
            write(Path(tmp, folder, "s.csv"), num_non_function_words=[9] * 10,
                  length_tokens=rng.normal(size=10), expertise_llm_score=rng.normal(size=10))
        keep, _ = assembly_correlations(Path(tmp), "keep", "base")
        drop, _ = assembly_correlations(Path(tmp), "drop", "base")
        irish = pd.concat([pd.read_csv(Path(tmp, f, "s.csv")) for f in ("8th_amendment", "gendereq")])
        every = pd.concat([irish, pd.read_csv(Path(tmp, "scotland", "s.csv"))])
    llm, length = "Expertise Llm Score", "Length Tokens"
    assert np.isclose(drop.loc[llm, length], irish["expertise_llm_score"].corr(irish["length_tokens"]))
    assert np.isclose(keep.loc[llm, length], every["expertise_llm_score"].corr(every["length_tokens"]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

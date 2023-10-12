import pandas as pd
from pytest import approx

from syndiffix.synthesizer import Synthesizer

from .conftest import *


def test_noiseless_category_category_dataset() -> None:
    raw_data = pd.DataFrame([["x", True], ["x", False], ["y", True]] * 15)
    syn_data = Synthesizer(raw_data, anonymization_context=NOISELESS_CONTEXT).sample()
    assert raw_data.value_counts().to_list() == syn_data.value_counts().to_list()


def test_noisy_category_numeric_dataset() -> None:
    raw_data = pd.DataFrame(
        [
            [True, 0],
            [True, 6],
            [True, 12],
            [True, 7],
            [True, 21],
            [True, 12],
            [True, 17],
            [True, 31],
            [False, 5],
            [False, 2],
            [False, 7],
            [False, 19],
            [False, 24],
            [False, 16],
            [False, 8],
            [False, 2],
            [True, 16],
            [True, 30],
            [False, 4],
            [False, 3],
            [False, 0],
        ]
    )
    syn_data = Synthesizer(raw_data, anonymization_context=NOISELESS_CONTEXT).sample()

    # Test categorical column.
    raw_category_counts = raw_data[0].value_counts()
    syn_category_counts = syn_data[0].value_counts()
    assert len(raw_category_counts) == len(syn_category_counts)
    assert syn_category_counts[True] == approx(raw_category_counts[True], abs=5)
    assert syn_category_counts[False] == approx(raw_category_counts[False], abs=5)

    # Test numeric column.
    assert syn_data[1].mean() == approx(raw_data[1].mean(), abs=5)
    assert syn_data[1].std() == approx(raw_data[1].std(), rel=0.2)


def test_string_ranges() -> None:
    raw_data = pd.DataFrame(
        [
            "Leopoldstraße 92",
            "Leopoldstraße 1",
            "Leopoldstraße 108",
            "Leopoldstraße 27",
            "Leopoldstraße 34",
            "Leopoldstraße 15",
            "Leopoldstraße 1",
            "Leopoldstraße 27",
            "Potsdamer Straße 47",
            "Potsdamer Straße 17",
            "Potsdamer Straße 2",
            "Potsdamer Straße 31",
            "Potsdamer Straße 2",
            "Potsdamer Straße 17",
            "Potsdamer Straße 2",
            "Potsdamer Straße 17",
            "Spandauer Str. 84",
            "Spandauer Str. 4",
            "Spandauer Str. 1",
            "Spandauer Str. 21",
            "Spandauer Str. 41",
            "Spandauer Str. 40",
            "Spandauer Str. 44",
            "Spandauer Str. 49",
            "Gerichtstraße 3",
            "Gerichtstraße 1",
            "Gerichtstraße 2a",
            "Gerichtstraße 2b",
            "Gerichtstraße 20",
            "Gerichtstraße 12",
            "Gerichtstraße 9",
            "Gerichtstraße 4",
        ]
    )
    syn_data = Synthesizer(raw_data, anonymization_context=NOISELESS_CONTEXT).sample()

    assert len(syn_data) == approx(len(raw_data), rel=0.1)

    syn_prefixes = set()
    for value in syn_data[0]:
        syn_prefixes.add(value[: value.find("*")])
    assert syn_prefixes.issuperset(["Leopoldstraße ", "Potsdamer Straße ", "Spandauer Str. 4", "Gerichtstraße "])

import numpy as np

from syndiffix.clustering.measures import measure_all

from ..conftest import *


def test_measure_all() -> None:
    forest = load_forest(
        "taxi-1000.csv",
        columns=["pickup_longitude", "pickup_latitude", "fare_amount", "rate_code", "passenger_count"],
        anon_params=NOISELESS_PARAMS,
    )

    measures = measure_all(forest)

    # Assert consistency with F# implementation.
    assert np.array_equal(
        np.round(measures.dependency_matrix, 2),
        np.array(
            [
                [1.00, 0.2, 0.15, 0.02, 0.05],
                [0.2, 1.00, 0.14, 0.02, 0.04],
                [0.15, 0.14, 1.00, 0.04, 0.08],
                [0.02, 0.02, 0.04, 1.00, 0.01],
                [0.05, 0.04, 0.08, 0.01, 1.00],
            ]
        ),
    )

    print(np.round(measures.entropy_1dim, 3))
    assert np.array_equal(np.round(measures.entropy_1dim, 3), np.array([9.218, 9.212, 5.164, 0.118, 1.350]))

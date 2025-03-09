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
                [0.2, 1.00, 0.18, 0.02, 0.05],
                [0.15, 0.18, 1.00, 0.04, 0.07],
                [0.02, 0.02, 0.04, 1.00, 0],
                [0.05, 0.05, 0.07, 0, 1.00],
            ]
        ),
    )

    assert np.array_equal(
        np.round(measures.entropy_1dim, 3),
        np.array([16.042, 15.886, 5.398, 0.118, 1.350]),
    )

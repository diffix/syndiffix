from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.feature_selection import RFECV, VarianceThreshold  # type: ignore
from sklearn.preprocessing import (  # type: ignore
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # type: ignore

# ----------------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------------


CATEGORY_THRESHOLD = 15


def _is_categorical(column: pd.Series) -> bool:
    return column.dtype == "object" or column.nunique() <= CATEGORY_THRESHOLD


def _get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    text_features = []
    continuous_features = []
    categorical_features = []

    for colname in df.columns:
        column = df[colname]
        nunique = column.nunique()
        if nunique <= CATEGORY_THRESHOLD:
            categorical_features.append(colname)
        elif column.dtype == "object":
            text_features.append(colname)
        else:
            continuous_features.append(colname)

    return text_features, continuous_features, categorical_features


def _one_hot_feature_names(encoder: Any, categorical_features: list[str]) -> tuple[list[str], dict[str, str]]:
    cats = [encoder._compute_transformed_categories(i) for i, _ in enumerate(encoder.categories_)]

    feature_names = []
    inverse_names = {}

    for i in range(len(cats)):
        category = categorical_features[i]
        names = [category + "$$" + str(label) for label in cats[i]]
        feature_names.extend(names)

        for name in names:
            inverse_names[name] = category

    return feature_names, inverse_names


def _random_encoder(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for column in X.columns:
        column_text = X[column].apply(str)
        unique_values = column_text.unique()
        np.random.shuffle(unique_values)
        mapping = {value: index for index, value in enumerate(unique_values)}
        X[column] = column_text.map(mapping).astype(int)

    return X


def _preprocess(
    df: pd.DataFrame, one_hot_encode: bool = True, variance_threshold: bool = True
) -> tuple[pd.DataFrame, dict[str, str]]:
    # This conversion needed because the blob code works with the original data
    for col in df.select_dtypes(include=["datetime64"]).columns:
        df[col] = df[col].astype(np.int64) // 10**9  # Convert to Unix timestamp
    text_features, continuous_features, categorical_features = _get_feature_types(df)
    ordinal_features = []

    if not one_hot_encode:
        ordinal_features = categorical_features
        categorical_features = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("random_encode", FunctionTransformer(_random_encoder), text_features),
            ("ord", OrdinalEncoder(), ordinal_features),
            ("num", RobustScaler(), continuous_features),
            ("cat", OneHotEncoder(sparse_output=False), categorical_features),
        ],
        sparse_threshold=0,
    )

    result = preprocessor.fit_transform(df)

    inverse_lookup = {name: name for name in text_features + ordinal_features + continuous_features}

    categorical_names: list[str] = []

    if len(categorical_features) > 0:
        categorical_names, categorical_inverse = _one_hot_feature_names(
            preprocessor.named_transformers_["cat"], categorical_features
        )
        inverse_lookup.update(categorical_inverse)

    df_preprocessed = pd.DataFrame(
        result, columns=text_features + ordinal_features + continuous_features + categorical_names
    )

    if not variance_threshold:
        return df_preprocessed, inverse_lookup

    threshold = VarianceThreshold(0.00001)
    threshold.set_output(transform="pandas")
    try:
        df_filtered = threshold.fit_transform(df_preprocessed)
    except ValueError:
        # This should be a rare event. It happens when none of the features have
        # variance above the threshold.
        df_filtered = df_preprocessed

    return df_filtered, inverse_lookup


def _split(df: pd.DataFrame, column: str, one_hot_X: bool) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    df = df.dropna(axis=0)
    X, X_inv = _preprocess(df.drop(column, axis=1), one_hot_encode=one_hot_X)
    y, _ = _preprocess(df[[column]], one_hot_encode=False)
    return X, X_inv, y


@dataclass(frozen=True)
class FeatureSelectionResult:
    valid: bool
    features: list[str]
    k: int
    k_features: list[str]
    cumulative_score: list[float]
    cumulative_score_std: list[float]
    encoded_scores: dict


def select_features_ml(
    df: pd.DataFrame,
    column: str,
    classifier_model: Any | None = None,
    regressor_model: Any | None = None,
    one_hot_X: bool = False,
) -> FeatureSelectionResult:
    X, X_inv, y = _split(df, column, one_hot_X)

    if y.shape[0] == 0 or y.shape[1] == 0:
        return FeatureSelectionResult(
            valid=False,
            features=[],
            k=0,
            k_features=[],
            cumulative_score=[],
            cumulative_score_std=[],
            encoded_scores={
                "features": [],
                "k": 0,
                "k_features": [],
                "cumulative_score": [],
                "cumulative_score_std": [],
            },
        )

    if X.shape[1] == 1:
        return FeatureSelectionResult(
            valid=False,
            features=[X.columns[0]],
            k=1,
            k_features=[X.columns[0]],
            cumulative_score=[0.0],
            cumulative_score_std=[0.0],
            encoded_scores={
                "features": [X.columns[0]],
                "k": 1,
                "kFeatures": [X.columns[0]],
                "cumulative_score": [0.0],
                "cumulative_score_std": [0.0],
            },
        )

    if _is_categorical(y[column]):
        estimator = classifier_model or DecisionTreeClassifier(random_state=0)
    else:
        estimator = regressor_model or DecisionTreeRegressor(random_state=0)

    rfecv = RFECV(estimator=estimator)
    rfecv.fit(X, y)

    feature_ranks = rfecv.ranking_
    feature_names = X.columns.tolist()

    sorted_features = sorted(zip(feature_names, feature_ranks), key=lambda x: x[1])

    encoded_k = int(rfecv.n_features_)
    encoded_features = [name for name, _ in sorted_features]
    encoded_scores = rfecv.cv_results_["mean_test_score"].tolist()
    encoded_scores_std = rfecv.cv_results_["std_test_score"].tolist()

    decoded_k = 0
    decoded_features: list[str] = []
    decoded_scores: list[float] = []
    decoded_scores_std: list[float] = []

    for i, feature in enumerate(encoded_features):
        decoded_feature = X_inv[feature]
        if decoded_feature in decoded_features:
            if decoded_features[-1] == decoded_feature:
                decoded_scores[-1] = encoded_scores[i]
                decoded_scores_std[-1] = encoded_scores_std[i]
        else:
            decoded_features.append(decoded_feature)
            decoded_scores.append(encoded_scores[i])
            decoded_scores_std.append(encoded_scores_std[i])

        if i == encoded_k - 1:
            decoded_k = len(decoded_features)

    return FeatureSelectionResult(
        valid=True,
        features=decoded_features,
        k=decoded_k,
        k_features=decoded_features[:decoded_k],
        cumulative_score=decoded_scores,
        cumulative_score_std=decoded_scores_std,
        encoded_scores={
            "features": encoded_features,
            "k": encoded_k,
            "k_features": encoded_features[:encoded_k],
            "cumulative_score": encoded_scores,
            "cumulative_score_std": encoded_scores_std,
        },
    )

"""
Stage 4 preprocessing. Every transformer here is fit on one dataframe and
applied to another — never fit on the data it will be evaluated against.

Numeric: median imputation, then standardisation (population std, matching
sklearn's StandardScaler convention). A was_missing_<feature> indicator is
added for every feature that had at least one null in the FIT data.

Categorical: one-hot, with null mapped to its own '__NULL__' category rather
than imputed. A category absent from the fit data produces no matching
column at transform time, so an unseen category at apply time is correctly
all-zeros.
"""

import pandas as pd


class NumericPreprocessor:
    def __init__(self, features: list[str], add_indicators: bool = True):
        self.features = list(features)
        self.add_indicators = add_indicators

    def fit(self, df: pd.DataFrame) -> "NumericPreprocessor":
        self.medians_ = {}
        self.means_ = {}
        self.stds_ = {}
        self.indicator_features_ = []
        for f in self.features:
            col = df[f].astype(float)
            n_null = int(col.isna().sum())
            median = float(col.median())
            self.medians_[f] = median
            if n_null > 0 and self.add_indicators:
                self.indicator_features_.append(f)
            imputed = col.fillna(median)
            mean = float(imputed.mean())
            std = float(imputed.std(ddof=0))
            self.means_[f] = mean
            self.stds_[f] = std if std > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = {}
        for f in self.features:
            col = df[f].astype(float)
            imputed = col.fillna(self.medians_[f])
            data[f] = (imputed - self.means_[f]) / self.stds_[f]
        for f in self.indicator_features_:
            data[f"was_missing_{f}"] = df[f].isna().astype(int)
        return pd.DataFrame(data, index=df.index)

    @property
    def output_columns(self) -> list[str]:
        return list(self.features) + [f"was_missing_{f}" for f in self.indicator_features_]


class CategoricalPreprocessor:
    NULL_TOKEN = "__NULL__"

    def __init__(self, features: list[str]):
        self.features = list(features)

    def fit(self, df: pd.DataFrame) -> "CategoricalPreprocessor":
        self.categories_ = {}
        for f in self.features:
            vals = df[f].fillna(self.NULL_TOKEN).astype(str)
            self.categories_[f] = sorted(vals.unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = {}
        for f in self.features:
            vals = df[f].fillna(self.NULL_TOKEN).astype(str)
            for cat in self.categories_[f]:
                data[f"{f}__{cat}"] = (vals == cat).astype(int)
        return pd.DataFrame(data, index=df.index)

    @property
    def output_columns(self) -> list[str]:
        cols = []
        for f in self.features:
            cols.extend(f"{f}__{cat}" for cat in self.categories_[f])
        return cols


class FeaturePreprocessor:
    """Combines numeric and categorical preprocessing into one design matrix."""

    def __init__(self, numeric_features: list[str], categorical_features: list[str],
                 add_indicators: bool = True):
        self.numeric = NumericPreprocessor(numeric_features, add_indicators=add_indicators)
        self.categorical = CategoricalPreprocessor(categorical_features)

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        self.numeric.fit(df)
        self.categorical.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num_df = self.numeric.transform(df)
        cat_df = self.categorical.transform(df)
        return pd.concat([num_df, cat_df], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def output_columns(self) -> list[str]:
        return self.numeric.output_columns + self.categorical.output_columns

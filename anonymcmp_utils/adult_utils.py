from apt.utils.dataset_utils import get_adult_dataset_pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def binarlize_column(value):
    if value == 'Wife' or value == 'Husband':
        return "Wife_or_Husband"
    else:
        return "The_others"


def get_dataset_bin_relationship():
    (x_train, y_train), (x_test, y_test) = get_adult_dataset_pd()

    bin_column = 'relationship'
    x_train[bin_column] = x_train[bin_column].apply(binarlize_column)
    x_test[bin_column] = x_test[bin_column].apply(binarlize_column)
    x_test = x_test.astype(x_train.dtypes)

    return (x_train, y_train), (x_test, y_test)


def get_dataset_preprocessor(x_train, scaler=True):
    numeric_features = x_train.select_dtypes(['int64']).columns.to_list()
    categorical_features = x_train.select_dtypes(['object']).columns.to_list()

    if scaler:
        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                   ('scaler', StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[('hotencoder', OneHotEncoder(handle_unknown="ignore", sparse=False, drop="if_binary")),
                   ('scaler', StandardScaler())]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
        )

        categorical_transformer = Pipeline(
            steps=[('hotencoder', OneHotEncoder(handle_unknown="ignore", sparse=False, drop="if_binary"))]
        )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


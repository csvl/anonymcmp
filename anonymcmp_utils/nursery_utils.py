from apt.utils.dataset_utils import get_nursery_dataset_pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def get_dataset():
    (x_train, y_train), (x_test, y_test) = get_nursery_dataset_pd(test_set=0.5, transform_social=True)

    convert_dict = {'parents': object,
                    'has_nurs': object,
                    'form': object,
                    'children': int,
                    'housing': object,
                    'finance': object,
                    'social': object,
                    'health': object}

    x_train = x_train.astype(convert_dict)
    x_test = x_test.astype(convert_dict)

    return (x_train, y_train), (x_test, y_test)


def get_dataset_preprocessor(scaler=True):
    numeric_features = ['children']
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health"]

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


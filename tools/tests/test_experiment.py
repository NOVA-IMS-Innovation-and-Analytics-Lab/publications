"""
Test the experiment module.
"""

from collections import Counter

import pytest

from sklearn.datasets import make_classification
from sklearnext.over_sampling import SMOTE

from ..experiment import (
    UnderOverSampler,
    check_estimators,
    CLASSIFIERS_MAPPING,
    OVERSAMPLERS_MAPPING,
    generate_configuration
)

X, y = make_classification(random_state=0, n_samples=100, weights=[0.8, 0.2])


def test_under_oversampler():
    """Test the UnderOverSampler class."""
    underoversampler = UnderOverSampler(random_state=1, oversampler=SMOTE(), factor=2)
    
    # Test fit
    underoversampler.fit(X, y)
    assert len(underoversampler.sampling_strategy_) == 1
    assert set(underoversampler.sampling_strategy_.values()) == set([0]) 
    assert underoversampler.undersampler_.random_state == 1
    assert underoversampler.undersampler_.sampling_strategy == {0: 40, 1:10}
    assert underoversampler.oversampler_.sampling_strategy == {0: 80, 1:20}

    # Test sample
    X_resampled, y_resampled = underoversampler.fit_resample(X, y)
    assert X.shape == X_resampled.shape
    assert y.shape == y_resampled.shape
    assert Counter(y) == Counter(y_resampled)


@pytest.mark.parametrize('category,mapping', [
    ('basic', CLASSIFIERS_MAPPING),
    ('basic', OVERSAMPLERS_MAPPING),
    ('clustering', OVERSAMPLERS_MAPPING),
    ('scaled', OVERSAMPLERS_MAPPING),
    ('undersampled', OVERSAMPLERS_MAPPING)
])
def test_default_check_estimators(category, mapping):
    """Test the default generation of estimators."""
    estimators = check_estimators(category, None, mapping)
    expected_estimators = mapping[category]
    for estimator, expected_estimator in zip(estimators, expected_estimators):
        assert estimator[0] == expected_estimator[0]
        assert estimator[1].__class__ == expected_estimator[1].__class__
        assert estimator[2] == expected_estimator[2]



@pytest.mark.parametrize('category,names,mapping', [
    ('basic', ['KNN', 'LR'], CLASSIFIERS_MAPPING),
    ('basic', ['NO OVERSAMPLING', 'RANDOM OVERSAMPLING', 'SMOTE'], OVERSAMPLERS_MAPPING),
    ('clustering', ['K-MEANS SMOTE'], OVERSAMPLERS_MAPPING),
    ('scaled', ['ADASYN', 'SMOTE'], OVERSAMPLERS_MAPPING),
    ('undersampled', ['BENCHMARK METHOD', 'NO OVERSAMPLING'], OVERSAMPLERS_MAPPING)
])
def test_subset_check_estimators(category, names, mapping):
    """Test the generation of estimators."""
    estimators = check_estimators(category, names, mapping)
    expected_estimators = [est for est in mapping[category] if est[0] in names]
    for estimator, expected_estimator in zip(estimators, expected_estimators):
        assert estimator[0] == expected_estimator[0]
        assert estimator[1].__class__ == expected_estimator[1].__class__
        assert estimator[2] == expected_estimator[2]


@pytest.mark.parametrize('category,names,mapping', [
    ('Basic', None, CLASSIFIERS_MAPPING),
    ('basic', ['knn'], CLASSIFIERS_MAPPING),
    ('Clustering', ['K-MEANS SMOTE'], OVERSAMPLERS_MAPPING),
    ('scaled', ['ADASYN', 'smote'], OVERSAMPLERS_MAPPING),
    ('undersampled', 'BENCHMARK METHOD', OVERSAMPLERS_MAPPING)
])
def test_check_estimators_raise_error(category, names, mapping):
    """Test the raise of error for false input."""
    with pytest.raises(ValueError):
        estimators = check_estimators(category, names, mapping)

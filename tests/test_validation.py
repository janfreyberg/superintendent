import pytest

pytestmark = pytest.mark.skip

# from sklearn.linear_model import LogisticRegression
# from superintendent.validation import valid_classifier  # , valid_data


# def test_valid_classifier():
#     assert valid_classifier(None) is None
#     dummy_classifier = LogisticRegression()
#     assert valid_classifier(dummy_classifier) is dummy_classifier
#     with pytest.raises(ValueError):
#         valid_classifier("dummy-non-classifier")


# def test_valid_data():
#     assert valid_classifier(None) is None
#     dummy_classifier = LogisticRegression()
#     assert valid_classifier(dummy_classifier) is dummy_classifier
#     with pytest.raises(ValueError):
#         valid_classifier("dummy-non-classifier")

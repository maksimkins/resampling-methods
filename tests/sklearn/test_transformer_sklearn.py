from sklearn.base import clone

from imblearn_resc.preprocessing import ReSCTransformer


def test_transformer_supports_sklearn_parameter_and_clone_protocol():
    transformer = ReSCTransformer()

    assert transformer.get_params(deep=True) == {}
    assert isinstance(clone(transformer), ReSCTransformer)

from src.models.my_model import MyModel


def test_model_name():
    model = MyModel(name="modelo_prueba")
    assert model.name == "modelo_prueba"


def test_model_predict_returns_metadata():
    model = MyModel(name="modelo_prueba")
    result = model.predict([1, 2, 3])
    assert result["model"] == "modelo_prueba"
    assert result["input_size"] == 3

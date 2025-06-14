import pytest
import numpy as np
from dash import html
from dash.exceptions import PreventUpdate
from dashapp import predict_diabetes

@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    class MockModel:
        def predict(self, X):
            return [1]  
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])  

    monkeypatch.setattr("dashapp.rf_model", MockModel())

def test_predict_diabetes_high_glucose(monkeypatch):

    answers = [1, 110, 70, 20, 79, 25.0, 0.5, 30]  # Glucose = 110
    result = predict_diabetes(n_clicks=1, answers=answers, current_step=7)

    assert isinstance(result[0], html.Div)
    assert "High Risk Detected" in result[0].children[0].children

def test_predict_diabetes_model_prediction(monkeypatch):

    answers = [1, 90, 70, 20, 79, 25.0, 0.5, 30]  # Glucose = 90
    result = predict_diabetes(n_clicks=1, answers=answers, current_step=7)

    assert "The model predicts diabetes" in result[0].children[0].children

def test_predict_no_diabetes(monkeypatch):

    class MockModel:
        def predict(self, X): return [0]  # no diabetes
        def predict_proba(self, X): return np.array([[0.9, 0.1]])

    monkeypatch.setattr("dashapp.rf_model", MockModel())

    answers = [1, 90, 70, 20, 79, 25.0, 0.5, 30]
    result = predict_diabetes(n_clicks=1, answers=answers, current_step=7)

    assert "The model predicts no diabetes" in result[0].children[0].children

def test_predict_invalid_nclicks():
    from dashapp import predict_diabetes
    with pytest.raises(PreventUpdate):
        predict_diabetes(n_clicks=None, answers=[0]*8, current_step=7)

def test_predict_diabetes_invalid_step():

    with pytest.raises(PreventUpdate):
        predict_diabetes(n_clicks=1, answers=[0]*8, current_step=5)

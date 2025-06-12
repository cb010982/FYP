import pytest
from login import handle_login_logic
from dashapp import predict_diabetes
from unittest.mock import patch
import dash
import numpy as np

@pytest.fixture
def mock_login_session(monkeypatch):

    fake_users = [{
        "user_id": 101,
        "name": "MockUser",
        "password_hash": "$2b$12$DummyHashJustForMocking"
    }]

    class MockDB:
        def execute(self, query, params=None):
            if "SELECT user_id, password_hash FROM users" in str(query):
                class Result:
                    user_id = fake_users[0]["user_id"]
                    password_hash = fake_users[0]["password_hash"]
                class FetchOne:
                    def fetchone(self_inner): return Result()
                return FetchOne()
            if "INSERT INTO sugar_levels" in str(query):
                return None
        def commit(self): pass
        def close(self): pass


    monkeypatch.setattr("login.SessionLocal", lambda: MockDB())


    from login import bcrypt
    monkeypatch.setattr(bcrypt, "checkpw", lambda p, h: True)

def test_login_then_predict(mock_login_session):
    # Step 1: Login 
    result = handle_login_logic("MockUser", "somepass", 120, 25.0)
    assert result[0] == "/dashapp1"
    assert result[1] == 101

    # Step 2: Predict diabetes 
    answers = [1, 90, 70, 20, 85, 25.0, 0.5, 30]  
    current_step = 7
    n_clicks = 1

    with patch("dashapp.rf_model") as mock_model:
        mock_model.predict.return_value = [1]  # diabetic
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]]) 

        result = predict_diabetes(n_clicks, answers, current_step)

     
        if len(result) == 3:
            output, glucose, bmi = result
        else:
            output, glucose = result
            bmi = None

        assert isinstance(output, dash.development.base_component.Component)
        assert "The model predicts diabetes" in output.children[0].children
        assert glucose == 90
        assert bmi == 25.0

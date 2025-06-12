from types import SimpleNamespace
import numpy as np
import pytest
import login
from signup import handle_signup_logic
from login import handle_login_logic
from dashapp import predict_diabetes
from dashapp1 import handle_rating
from unittest.mock import patch
import pandas as pd


def test_full_user_journey(monkeypatch):
    # STEP 1: SIGNUP
    name = "FunctionalUser"
    email = "func@example.com"
    password = "securepass"
    sugar = 130
    bmi = 25.0

    monkeypatch.setattr("signup.SessionLocal", lambda: type("MockDB", (), {
        "execute": lambda self, query, params=None: type("Result", (), {
            "fetchone": lambda self: None, 
            "scalar": lambda self: 100  
        })(),
        "commit": lambda self: None,
        "close": lambda self: None
    })())

    result = handle_signup_logic(name, email, password, sugar, bmi)
    assert result[0] == "/dashapp1"

    # STEP 2: LOGIN

    class MockLoginResult:
        def fetchone(self):
            return SimpleNamespace(
                user_id=101,
                password_hash="$2b$12$Dummy"  
            )


    monkeypatch.setattr("login.SessionLocal", lambda: type("MockDB", (), {
        "execute": lambda self, q, p=None: MockLoginResult(), 
        "commit": lambda self: None,
        "close": lambda self: None
    })())

    monkeypatch.setattr(login.bcrypt, "checkpw", lambda p, h: True)

    result = handle_login_logic(name, password, sugar, bmi)
    assert result[0] == "/dashapp1"

    # STEP 3: PREDICTION
    answers = [1, 90, 70, 20, 85, 25.0, 0.5, 30]
    class MockModel:
        def predict(self, x): return [1]
        def predict_proba(self, x): return np.array([[0.2, 0.8]])

    monkeypatch.setattr("dashapp.rf_model", MockModel())

    result = predict_diabetes(n_clicks=1, answers=answers, current_step=7)
    assert "predicts diabetes" in result[0].children[0].children

    # STEP 4: RECOMMENDATION FEEDBACK
    user_id = 101
    course_id = 999

    monkeypatch.setattr("dashapp1.course_df", pd.DataFrame([{
        "course_id": course_id,
        "course_name": "Mock Meal",
        "category": "main-dish",
        "image_url": "",
        "ingredients": "",
        "cooking_directions": "",
        "calories": 300,
        "sugar": 10,
        "fiber": 5
    }]))
    monkeypatch.setattr("dashapp1.scaler", type("MockScaler", (), {
        "transform": lambda self, x: [[0.5, 0.2, 0.1]]
    })())
    monkeypatch.setattr("dashapp1.SessionLocal", lambda: type("MockDB", (), {
        "execute": lambda self, q, p=None: type("R", (), {
            "fetchone": lambda self: ("MockUser",),
            "fetchall": lambda self: [(user_id, 0)],
            "scalar": lambda self: 1
        })(),
        "commit": lambda self: None,
        "close": lambda self: None
    })())
    monkeypatch.setattr("dashapp1.ctx", type("ctx", (), {
        "inputs_list": [
            [  # like buttons
                {"id": {"type": "like-btn", "index": 998}},   
                {"id": {"type": "like-btn", "index": course_id}}  
            ],
            [  # dislike buttons
                {"id": {"type": "dislike-btn", "index": 997}},
                {"id": {"type": "dislike-btn", "index": 996}}
            ]
        ],
        "triggered_id": {"type": "like-btn", "index": course_id}
    }))

    result = handle_rating(
        likes=[None, 1],
        dislikes=[None, None],
        like_times=[None, 1000000],
        dislike_times=[None, None],
        user_id=user_id
    )
    assert result.startswith(" Feedback recorded for course")

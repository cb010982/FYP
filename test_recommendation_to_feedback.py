from dashapp1 import handle_rating
from unittest.mock import patch
import dash
import pytest
import pandas as pd
import datetime

def test_recommendation_to_feedback(monkeypatch):
    user_id = 101
    course_id = 999

    likes = [None, 1]
    dislikes = [None, None]
    like_times = [None, 1000000]
    dislike_times = [None, None]

    monkeypatch.setattr("dashapp1.course_df", pd.DataFrame([{
        "course_id": course_id,
        "course_name": "Mock Meal",
        "category": "main-dish",
        "image_url": "http://example.com/image.jpg",
        "ingredients": "mock ingredients",
        "cooking_directions": "mock directions",
        "calories": 300,
        "sugar": 10,
        "fiber": 5
    }]))

    monkeypatch.setattr("dashapp1.scaler", lambda: None)
    class MockScaler:
        def transform(self, X): return [[0.5, 0.2, 0.1]]
    monkeypatch.setattr("dashapp1.scaler", MockScaler())

    class MockDB:
        def execute(self, query, params=None):
            class Result:
                def fetchall(self_inner): return [(user_id, 0)]
                def scalar(self_inner): return 1
                def fetchone(self_inner): return ("MockUser",)  
            return Result()

        def commit(self): pass
        def close(self): pass

    monkeypatch.setattr("dashapp1.SessionLocal", lambda: MockDB())

    monkeypatch.setattr("dashapp1.ctx", type("ctx", (), {
        "inputs_list": [
            [  # likes
                {"id": {"type": "like-btn", "index": 998}},
                {"id": {"type": "like-btn", "index": course_id}} 
            ],
            [  # dislikes
                {"id": {"type": "dislike-btn", "index": 997}},
                {"id": {"type": "dislike-btn", "index": 996}}
            ]
        ],
        "triggered_id": {"type": "like-btn", "index": course_id}
    }))


    result = handle_rating(likes, dislikes, like_times, dislike_times, user_id)
    assert result.startswith(" Feedback recorded for course")

import pytest
import torch
import pandas as pd
from recommendation_logic import get_filtered_recommendations, classify_sugar_level

def test_classify_sugar_level():
    assert classify_sugar_level(90) == "low"
    assert classify_sugar_level(140) == "normal"
    assert classify_sugar_level(200) == "high"

def test_get_filtered_recommendations_with_mock_model(monkeypatch):
    # Mock model
    class MockItemEmbedding:
        num_embeddings = 10

    class MockModel:
        item_embedding = MockItemEmbedding()
        def __call__(self, user, item):
            return torch.rand(len(item))

    # Dummy course data
    dummy_courses = pd.DataFrame({
        "course_id": [str(i) for i in range(10)],
        "sugar": [10]*10,
        "calories": [300]*10,
        "fiber": [3]*10,
        "course_name": ["meal"]*10,
        "category": ["main-dish"]*10,
        "image_url": ["url"]*10,
        "ingredients": ["ing"]*10,
        "cooking_directions": ["step"]*10
    })

    # Mock DB session 
    class MockResult:
        def scalar(self): return 0
        def fetchall(self): return [(0, 0), (1, 1)]  # Disliked 0, Liked 1

    class MockDB:
        def execute(self, query, params=None): return MockResult()
        def close(self): pass

    monkeypatch.setattr("recommendation_logic.SessionLocal", lambda: MockDB())

    # New user with high BMI
    result = get_filtered_recommendations(
        user_index=None,
        sugar_value=150,
        bmi_value=31,
        is_new_user=True,
        course_df=dummy_courses,
        model=MockModel(),
        num_items=10,
        top_n=5
    )
    assert not result.empty
    assert len(result) <= 5

    result2 = get_filtered_recommendations(
        user_index=0,
        sugar_value=150,
        bmi_value=25,
        is_new_user=False,
        course_df=dummy_courses,
        model=MockModel(),
        num_items=10,
        top_n=10
    )
    assert not result2.empty
    assert "0" not in result2["course_id"].values  
    assert "1" in result2["course_id"].values      
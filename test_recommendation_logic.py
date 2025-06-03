import pytest
import torch
import pandas as pd
from recommendation_logic import get_filtered_recommendations, classify_sugar_level

def test_classify_sugar_level():
    assert classify_sugar_level(90) == "low"
    assert classify_sugar_level(140) == "normal"
    assert classify_sugar_level(200) == "high"

def test_get_filtered_recommendations_with_mock_model():
    class MockModel:
        def __call__(self, user, item): return torch.rand(len(item))

    dummy_courses = pd.DataFrame({
        "course_id": range(10),
        "sugar": [10]*10,
        "calories": [300]*10,
        "fiber": [3]*10,
        "course_name": ["meal"]*10,
        "category": ["main-dish"]*10,
        "image_url": ["url"]*10,
        "ingredients": ["ing"]*10,
        "cooking_directions": ["step"]*10
    })

    filtered = get_filtered_recommendations(
        user_index=0,
        sugar_value=150,
        bmi_value=28,
        is_new_user=False,
        course_df=dummy_courses,
        model=MockModel(),
        num_items=10,
        top_n=5
    )

    assert not filtered.empty
    assert len(filtered) <= 5

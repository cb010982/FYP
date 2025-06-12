import pytest
from dashapp1 import display_recommendations
from unittest.mock import patch
from dash import html

@pytest.fixture
def mock_user_mapping(monkeypatch):
    from sqlalchemy.engine import ResultProxy
    import pandas as pd

    fake_user_id = 101
    fake_user_index = 9
    monkeypatch.setattr("dashapp1.SessionLocal", lambda: DummyDB())

    class DummyDB:
        def execute(self, query, params=None):
            if "filtered_df" in str(query):
                return DummyResult([
                    (fake_user_id, fake_user_index)
                ])
            elif "sugar_levels" in str(query):
                return DummyResult([
                    (120, "2024-06-05 12:00:00")
                ])
        def close(self): pass

    class DummyResult:
        def __init__(self, rows): self.rows = rows
        def fetchall(self): return self.rows

def test_display_recommendations(mock_user_mapping):
    user_id = 101
    sugar_value = 120
    bmi_value = 25.0

    output = display_recommendations(user_id, sugar_value, bmi_value)

    assert isinstance(output, html.Div)
    assert any("carousel-wrapper" in str(child) for child in output.children)

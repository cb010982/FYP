import dash
import pytest
import bcrypt
from unittest.mock import MagicMock
from signup import handle_signup_logic

@pytest.fixture
def mock_db(monkeypatch):
    mock_session = MagicMock()
    monkeypatch.setattr("signup.SessionLocal", lambda: mock_session)
    return mock_session

def test_signup_success(monkeypatch, mock_db):
    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: None),       
        MagicMock(scalar=lambda: 1),             
        None,
        None,  
    ]
    mock_db.commit = MagicMock()

    result = handle_signup_logic(
        name="Test User",
        email="test@example.com",
        password="password123",
        sugar=110,
        bmi_value=22.5
    )

    assert result[0] == "/dashapp1"
    assert result[1] == "2" 
    assert result[2] == "Test User"
    assert result[3] == "110.0"
    assert result[4] == "22.5"


def test_signup_duplicate_email(monkeypatch, mock_db):
    mock_db.execute.side_effect = [
        MagicMock(fetchone=lambda: True), 
    ]
    result = handle_signup_logic("User", "test@example.com", "pass", 100, 20)
    assert result == (dash.no_update, None, None, None, None)

def test_signup_missing_fields():
    result = handle_signup_logic("", "email@test.com", "pass", 100, 20)
    assert result == (dash.no_update, None, None, None, None)

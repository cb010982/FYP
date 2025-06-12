import pytest
import bcrypt
from unittest.mock import MagicMock
from login import handle_login_logic  

@pytest.fixture
def mock_db_session(monkeypatch):

    mock_session = MagicMock()
    monkeypatch.setattr("login.SessionLocal", lambda: mock_session)
    return mock_session

def test_process_login_success(monkeypatch, mock_db_session):

    password = "password123"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    mock_user = type("User", (), {"user_id": 1, "password_hash": hashed_pw})()
    def mock_execute(query, params):
        class Result:
            def fetchone(self_inner): return mock_user
        return Result()

    mock_db_session.execute.side_effect = mock_execute
    result = handle_login_logic("testuser", password, 110, 22.5)
    assert result[0] == "/dashapp1"
    assert result[1] == 1
    assert result[2] == "testuser"
    assert result[3] == 110
    assert result[4] == 22.5
    assert result[5] is None

def test_process_login_failure(monkeypatch, mock_db_session):

    def mock_execute(query, params):
        class Result:
            def fetchone(self_inner): return None
        return Result()

    mock_db_session.execute.side_effect = mock_execute

    result = handle_login_logic("unknown", "wrongpass", 110, 20)

    assert "Invalid username or password" in result[5]

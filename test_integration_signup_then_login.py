import pytest
from unittest.mock import MagicMock
from signup import handle_signup_logic
from login import handle_login_logic
from sqlalchemy import text

@pytest.fixture
def mock_session(monkeypatch):
    from signup import SessionLocal as SignupSession
    from login import SessionLocal as LoginSession


    fake_users = []
    fake_sugar_levels = []

    
    mock_db = MagicMock()

    def execute(query, params=None):
        class MockResult:
            def fetchone(self):
                if "SELECT 1 FROM users" in str(query):
                    for user in fake_users:
                        if user["email"] == params["email"]:
                            return True
                    return None
                return None

            def scalar(self):
                if "SELECT MAX(user_id)" in str(query):
                    return max((user["user_id"] for user in fake_users), default=None)
                return None

        if "SELECT 1 FROM users" in str(query):
            return MockResult()

        if "SELECT MAX(user_id)" in str(query):
            return MockResult()

        if "INSERT INTO users" in str(query):
            fake_users.append({
                "user_id": params["user_id"],
                "email": params["email"],
                "name": params["name"],
                "password_hash": params["password_hash"],
            })
            return None

        if "INSERT INTO sugar_levels" in str(query):
            fake_sugar_levels.append({
                "user_id": params["user_id"],
                "sugar": params["sugar"]
            })
            return None

        if "SELECT user_id, password_hash FROM users" in str(query):
            class LoginResult:
                def fetchone(self_inner):
                    for user in fake_users:
                        if user["name"] == params["username"]:
                            class Result:
                                user_id = user["user_id"]
                                password_hash = user["password_hash"]
                            return Result()
                    return None
            return LoginResult()

    mock_db.execute.side_effect = execute
    mock_db.commit.return_value = None
    mock_db.close.return_value = None

   
    monkeypatch.setattr("signup.SessionLocal", lambda: mock_db)
    monkeypatch.setattr("login.SessionLocal", lambda: mock_db)

    return mock_db

def test_signup_then_login(mock_session):
    name = "MockUser"
    email = "mock@example.com"
    password = "secure123"
    sugar = 120
    bmi_value = 23.5

    from bcrypt import hashpw, gensalt
    hashed_password = hashpw(password.encode(), gensalt()).decode()

    # Step 1: Sign up
    signup_result = handle_signup_logic(name, email, password, sugar, bmi_value)
    assert signup_result[0] == "/dashapp1"
    user_id = int(signup_result[1])

    # Step 2: Login

    from login import bcrypt
    mock_session  
    bcrypt.checkpw = lambda p, h: True


    login_result = handle_login_logic(name, password, sugar, bmi_value)
    assert login_result[0] == "/dashapp1"
    assert login_result[1] == user_id
    assert login_result[2] == name
    assert login_result[3] == sugar
    assert login_result[4] == bmi_value
    assert login_result[5] is None

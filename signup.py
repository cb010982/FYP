
# import dash
# from dash import html, dcc
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
# import sqlite3
# from sqlalchemy import text
# from db import SessionLocal
# import bcrypt
# from dash.exceptions import PreventUpdate

# layout = html.Div([
#     dcc.Store(id="session-user-id", storage_type="session"),
#     dcc.Store(id="session-username", storage_type="session"),
#     dcc.Store(id="session-sugar", storage_type="session"),
#     dcc.Store(id="session-bmi", storage_type="session"), #bmi store
#     dcc.Store(id="temp-glucose-value", storage_type="session"),  # Add this to the stores at the top of the layout
#     dcc.Store(id="temp-bmi-value", storage_type="session"),  # NEW BMI store


#     dcc.Location(id='signup-url', refresh=True),  # Add Location component
#     dbc.Row([
#         # LEFT SIDE
#         dbc.Col([
#             html.Div([
#                 html.Div(className="auth-back-link"),
#                 html.Div([
#                     html.H2("Sign Up", className="auth-title"),
#                     html.P("Join us and start your journey today!", className="auth-subtitle"),
                    
#                     dbc.Form([
#                         dbc.Label("Name *", className="auth-label"),
#                         dbc.Input(id="signup-name", type="text", placeholder="Your Name", className="mb-3"),
                        
#                         dbc.Label("Email *", className="auth-label"),
#                         dbc.Input(id="signup-email", type="email", placeholder="mail@example.com", className="mb-3"),
                        
#                         dbc.Label("Password *", className="auth-label"),
#                         dbc.Input(id="signup-password", type="password", placeholder="Min. 8 characters", className="mb-3"),

#                         dbc.Label([
#                             "Current Sugar Level *"
#                         ], className="auth-label"),
#                         dbc.Input(
#                             id="sugar",
#                             type="number",
#                             placeholder="e.g. 110",
#                             min=50,
#                             max=300,
#                             step=1,
#                             className="mb-3"
#                         ),
                        
#                         dbc.Button("Sign Up", id="signup-button", className="w-100 auth-button")
#                     ]),
                    
#                     html.P([
#                         "Already have an account? ",
#                         dcc.Link("Login", href="/login", className="auth-link")
#                     ], className="mt-3 text-center")
#                 ], className="auth-form")
#             ], className="auth-form-container")
#         ], width=6),
        
#         # RIGHT SIDE
#         dbc.Col([], width=6, className="auth-image-side signup-image")
#     ], className="g-0")
# ], className="auth-container")

# def init_callbacks(app):
#     @app.callback(
#         Output("signup-url", "pathname", allow_duplicate=True),
#         Output("session-user-id", "data", allow_duplicate=True),
#         Output("session-username", "data", allow_duplicate=True),
#         Output("session-sugar", "data", allow_duplicate=True),
#         Output("session-bmi", "data", allow_duplicate=True),  # Added this line for BMI
#         Input("signup-button", "n_clicks"),
#         State("signup-name", "value"),
#         State("signup-email", "value"),
#         State("signup-password", "value"),
#         State("sugar", "value"),
#         State("temp-bmi-value", "data"), # Added for BMI

#         prevent_initial_call=True
#     )
#     def process_signup(n_clicks, name, email, password, sugar, bmi_value):
#         if not all([email, password, name]):
#             raise PreventUpdate

#         db = SessionLocal()
#         try:
#             print("🚀 Sign-up callback triggered!")

#             # 1. Check if user already exists
#             query = text("SELECT 1 FROM users WHERE email = :email")
#             if db.execute(query, {"email": email}).fetchone():
#                 return dash.no_update, None, None, None, None

#             # 2. Hash the password
#             password_hash = bcrypt.hashpw(str(password).encode(), bcrypt.gensalt()).decode()

#             # 3. Get next user_id (manual assignment like in SQLite)
#             max_id_query = text("SELECT MAX(user_id) FROM users")
#             # last_id = db.execute(max_id_query).scalar()
#             # new_id = (last_id + 1) if last_id is not None else 0
#             last_id = db.execute(max_id_query).scalar()
#             new_id = (int(last_id) + 1) if last_id is not None else 0

            
#             # 4. Insert new user
#             insert_user_query = text("""
#                 INSERT INTO users (user_id, email, password_hash, name, created_at)
#                 VALUES (:user_id, :email, :password_hash, :name, NOW())
#             """)
#             db.execute(insert_user_query, {
#                 "user_id": new_id,
#                 "email": email,
#                 "password_hash": password_hash,
#                 "name": name
#             })

#             # 5. Insert sugar value
#             insert_sugar_query = text("""
#                 INSERT INTO sugar_levels (user_id, sugar_value)
#                 VALUES (:user_id, :sugar)
#             """)
#             db.execute(insert_sugar_query, {
#                 "user_id": new_id,
#                 "sugar": sugar
#             })

#             db.commit()
#             print(f"📦 Returning from signup: /dashapp1, {new_id}, {name}, {float(sugar)}, {float(bmi_value) if bmi_value else None}")

#             return "/dashapp1", str(new_id), str(name), str(float(sugar)), str(float(bmi_value)) if bmi_value else None

#         except Exception as e:
#             print("Signup error:", e)
#             return dash.no_update, None, None, None, None

#         finally:
#             db.close()


#     # Add this callback in the init_callbacks function
#     @app.callback(
#         Output("sugar", "value"),
#         Input("temp-glucose-value", "data")
#     )
    
#     def prefill_sugar_value(glucose_value):
#         if glucose_value is not None:
#             return glucose_value
#         return dash.no_update



import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import sqlite3
from sqlalchemy import text
from db import SessionLocal
import bcrypt
from dash.exceptions import PreventUpdate

layout = html.Div([
    dcc.Store(id="session-user-id", storage_type="session"),
    dcc.Store(id="session-username", storage_type="session"),
    dcc.Store(id="session-sugar", storage_type="session"),
    dcc.Store(id="session-bmi", storage_type="session"), #bmi store
    dcc.Store(id="temp-glucose-value", storage_type="session"),  # Add this to the stores at the top of the layout
    dcc.Store(id="temp-bmi-value", storage_type="session"),  # NEW BMI store


    dcc.Location(id='signup-url', refresh=True),  # Add Location component
    dbc.Row([
        # LEFT SIDE
        dbc.Col([
            html.Div([
                html.Div(className="auth-back-link"),
                html.Div([
                    html.H2("Sign Up", className="auth-title"),
                    html.P("Join us and start your journey today!", className="auth-subtitle"),
                    
                    dbc.Form([
                        dbc.Label("Name *", className="auth-label"),
                        dbc.Input(id="signup-name", type="text", placeholder="Your Name", className="mb-3"),
                        
                        dbc.Label("Email *", className="auth-label"),
                        dbc.Input(id="signup-email", type="email", placeholder="mail@example.com", className="mb-3"),
                        
                        dbc.Label("Password *", className="auth-label"),
                        dbc.Input(id="signup-password", type="password", placeholder="Min. 8 characters", className="mb-3"),

                        dbc.Label([
                            "Current Sugar Level *"
                        ], className="auth-label"),
                        dbc.Input(
                            id="sugar",
                            type="number",
                            placeholder="e.g. 110",
                            min=50,
                            max=300,
                            step=1,
                            className="mb-3"
                        ),
                        
                        dbc.Button("Sign Up", id="signup-button", className="w-100 auth-button")
                    ]),
                    
                    html.P([
                        "Already have an account? ",
                        dcc.Link("Login", href="/login", className="auth-link")
                    ], className="mt-3 text-center")
                ], className="auth-form")
            ], className="auth-form-container")
        ], width=6),
        
        # RIGHT SIDE
        dbc.Col([], width=6, className="auth-image-side signup-image")
    ], className="g-0")
], className="auth-container")


def handle_signup_logic(name, email, password, sugar, bmi_value):
    if not all([email, password, name, sugar]):
        return dash.no_update, None, None, None, None

    db = SessionLocal()
    try:
        # Check for duplicate
        query = text("SELECT 1 FROM users WHERE email = :email")
        if db.execute(query, {"email": email}).fetchone():
            return dash.no_update, None, None, None, None

        password_hash = bcrypt.hashpw(str(password).encode(), bcrypt.gensalt()).decode()

        last_id = db.execute(text("SELECT MAX(user_id) FROM users")).scalar()
        new_id = (int(last_id) + 1) if last_id is not None else 0

        db.execute(text("""
            INSERT INTO users (user_id, email, password_hash, name, created_at)
            VALUES (:user_id, :email, :password_hash, :name, NOW())
        """), {
            "user_id": new_id,
            "email": email,
            "password_hash": password_hash,
            "name": name
        })

        db.execute(text("""
            INSERT INTO sugar_levels (user_id, sugar_value)
            VALUES (:user_id, :sugar)
        """), {
            "user_id": new_id,
            "sugar": sugar
        })

        db.commit()
        return "/dashapp1", str(new_id), str(name), str(float(sugar)), str(float(bmi_value)) if bmi_value else None

    except Exception as e:
        print("Signup error:", e)
        return dash.no_update, None, None, None, None
    finally:
        db.close()

def init_callbacks(app):
    @app.callback(
        Output("signup-url", "pathname", allow_duplicate=True),
        Output("session-user-id", "data", allow_duplicate=True),
        Output("session-username", "data", allow_duplicate=True),
        Output("session-sugar", "data", allow_duplicate=True),
        Output("session-bmi", "data", allow_duplicate=True),
        Input("signup-button", "n_clicks"),
        State("signup-name", "value"),
        State("signup-email", "value"),
        State("signup-password", "value"),
        State("sugar", "value"),
        State("temp-bmi-value", "data"),
        prevent_initial_call=True
    )
    def process_signup(n_clicks, name, email, password, sugar, bmi_value):
        if not all([email, password, name]):
            raise PreventUpdate
        return handle_signup_logic(name, email, password, sugar, bmi_value)

    @app.callback(
        Output("sugar", "value"),
        Input("temp-glucose-value", "data")
    )
    def prefill_sugar_value(glucose_value):
        if glucose_value is not None:
            return glucose_value
        return dash.no_update

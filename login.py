
# import dash
# from dash import html, dcc
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
# from dash.exceptions import PreventUpdate
# import sqlite3
# import bcrypt
# from sqlalchemy import text
# from db import SessionLocal


# layout = html.Div([
#     dcc.Location(id='login-url', refresh=True),

#     # Session Stores
#     dcc.Store(id="session-user-id", storage_type="session"),
#     dcc.Store(id="session-username", storage_type="session"),
#     dcc.Store(id="session-sugar", storage_type="session"),
#     dcc.Store(id="temp-glucose-value", storage_type="session"),
#     dcc.Store(id="temp-bmi-value", storage_type="session"),  # bmi added as a new store

#     dbc.Row([
#         # LEFT SIDE
#         dbc.Col([
#             html.Div([
#                 html.Div(className="auth-back-link"),
#                 html.Div([
#                     html.H2("Login", className="auth-title"),
#                     html.P("Enter your details to get personalized meal suggestions", className="auth-subtitle"),

#                     dbc.Form([
#                         dbc.Label("Username *", className="auth-label"),
#                         dbc.Input(id="login-username", type="text", placeholder="Your Username", className="mb-3"),

#                         dbc.Label("Password *", className="auth-label"),
#                         dbc.Input(id="login-password", type="password", placeholder="Min. 8 characters", className="mb-3"),
#                         dcc.Store(id="session-bmi", storage_type="session"),  #  added for BMI

#                         dbc.Label([
#                             "Current Sugar Level *"
#                         ], className="auth-label"),
#                         dbc.Input(
#                             id="login-sugar",
#                             type="number",
#                             placeholder="e.g. 110",
#                             min=50,
#                             max=300,
#                             step=1,
#                             className="mb-3"
#                         ),

#                         dbc.Button("Login", id="login-button", color="primary", className="w-100 auth-button")
#                     ]),

#                     html.P([
#                         "Don't have an account? ",
#                         dcc.Link("Sign Up", href="/signup", className="auth-link")
#                     ], className="mt-3 text-center"),

#                     html.Div(id="login-error", className="text-danger mt-2")
#                 ], className="auth-form")
#             ], className="auth-form-container")
#         ], width=6),

#         # RIGHT SIDE
#         dbc.Col([], width=6, className="auth-image-side login-image")
#     ], className="g-0")
# ], className="auth-container")


# def init_callbacks(app):
#     @app.callback(
#         Output('login-url', 'pathname', allow_duplicate=True),
#         Output('session-user-id', 'data', allow_duplicate=True),
#         Output('session-username', 'data', allow_duplicate=True),
#         Output('session-sugar', 'data', allow_duplicate=True),
#         Output('session-bmi', 'data', allow_duplicate=True), #added for BMI
#         Output('login-error', 'children'),
#         Input('login-button', 'n_clicks'),
#         State('login-username', 'value'),
#         State('login-password', 'value'),
#         State('login-sugar', 'value'),
#         State('temp-bmi-value', 'data'), # Added for BMI
#         prevent_initial_call=True
#     )
#     def process_login(n_clicks, username, password, sugar, bmi_value):
#         if not all([username, password, sugar]):
#             return dash.no_update, None, None, None, None, " Please fill in all fields."

#         db = SessionLocal()
#         try:
#             # Secure raw SQL to avoid ORM complexity
#             query = text("SELECT user_id, password_hash FROM users WHERE name = :username")
#             result = db.execute(query, {"username": username}).fetchone()

#             if result and bcrypt.checkpw(password.encode(), result.password_hash.encode()):
#                 user_id = result.user_id

#                 # Insert sugar level
#                 insert_stmt = text("INSERT INTO sugar_levels (user_id, sugar_value) VALUES (:user_id, :sugar)")
#                 db.execute(insert_stmt, {"user_id": user_id, "sugar": sugar})
#                 db.commit()

#                 return "/dashapp1", user_id, username, sugar, bmi_value, None
#             else:
#                 return dash.no_update, None, None, None, None, " Invalid username or password."

#         except Exception as e:
#             return dash.no_update, None, None, None, None, f" Login failed: {str(e)}"

#         finally:
#             db.close()

#     @app.callback(
#         Output("login-sugar", "value"),
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
from dash.exceptions import PreventUpdate
import bcrypt
from sqlalchemy import text
from db import SessionLocal


# ======= UI Layout =======
layout = html.Div([
    dcc.Location(id='login-url', refresh=True),

    dcc.Store(id="session-user-id", storage_type="session"),
    dcc.Store(id="session-username", storage_type="session"),
    dcc.Store(id="session-sugar", storage_type="session"),
    dcc.Store(id="temp-glucose-value", storage_type="session"),
    dcc.Store(id="temp-bmi-value", storage_type="session"),
    dcc.Store(id="session-bmi", storage_type="session"),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div(className="auth-back-link"),
                html.Div([
                    html.H2("Login", className="auth-title"),
                    html.P("Enter your details to get personalized meal suggestions", className="auth-subtitle"),

                    dbc.Form([
                        dbc.Label("Username *", className="auth-label"),
                        dbc.Input(id="login-username", type="text", placeholder="Your Username", className="mb-3"),

                        dbc.Label("Password *", className="auth-label"),
                        dbc.Input(id="login-password", type="password", placeholder="Min. 8 characters", className="mb-3"),

                        dbc.Label("Current Sugar Level *", className="auth-label"),
                        dbc.Input(
                            id="login-sugar",
                            type="number",
                            placeholder="e.g. 110",
                            min=50,
                            max=300,
                            step=1,
                            className="mb-3"
                        ),

                        dbc.Button("Login", id="login-button", color="primary", className="w-100 auth-button")
                    ]),

                    html.P([
                        "Don't have an account? ",
                        dcc.Link("Sign Up", href="/signup", className="auth-link")
                    ], className="mt-3 text-center"),

                    html.Div(id="login-error", className="text-danger mt-2")
                ], className="auth-form")
            ], className="auth-form-container")
        ], width=6),

        dbc.Col([], width=6, className="auth-image-side login-image")
    ], className="g-0")
], className="auth-container")


# ======= Extracted Login Logic for Testing =======
def handle_login_logic(username, password, sugar, bmi_value):
    if not all([username, password, sugar]):
        return dash.no_update, None, None, None, None, " Please fill in all fields."

    db = SessionLocal()
    try:
        query = text("SELECT user_id, password_hash FROM users WHERE name = :username")
        result = db.execute(query, {"username": username}).fetchone()

        if result and bcrypt.checkpw(password.encode(), result.password_hash.encode()):
            user_id = result.user_id
            insert_stmt = text("INSERT INTO sugar_levels (user_id, sugar_value) VALUES (:user_id, :sugar)")
            db.execute(insert_stmt, {"user_id": user_id, "sugar": sugar})
            db.commit()
            return "/dashapp1", user_id, username, sugar, bmi_value, None
        else:
            return dash.no_update, None, None, None, None, " Invalid username or password."
    except Exception as e:
        return dash.no_update, None, None, None, None, f" Login failed: {str(e)}"
    finally:
        db.close()


# ======= Dash Callback Integration =======
def init_callbacks(app):
    @app.callback(
        Output('login-url', 'pathname', allow_duplicate=True),
        Output('session-user-id', 'data', allow_duplicate=True),
        Output('session-username', 'data', allow_duplicate=True),
        Output('session-sugar', 'data', allow_duplicate=True),
        Output('session-bmi', 'data', allow_duplicate=True),
        Output('login-error', 'children'),
        Input('login-button', 'n_clicks'),
        State('login-username', 'value'),
        State('login-password', 'value'),
        State('login-sugar', 'value'),
        State('temp-bmi-value', 'data'),
        prevent_initial_call=True
    )
    def process_login(n_clicks, username, password, sugar, bmi_value):
        return handle_login_logic(username, password, sugar, bmi_value)

    @app.callback(
        Output("login-sugar", "value"),
        Input("temp-glucose-value", "data")
    )
    def prefill_sugar_value(glucose_value):
        if glucose_value is not None:
            return glucose_value
        return dash.no_update

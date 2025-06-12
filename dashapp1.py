import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
import gdown
import pandas as pd
import torch
import torch.nn as nn
import sqlite3
import joblib
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from db import SessionLocal
from sqlalchemy import text
import os

def download_if_needed(filename, file_id):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

# Only when deployed on Railway
if os.environ.get("RAILWAY_ENVIRONMENT"):
    download_if_needed("filtered_df.csv", "1b4oobA7kibaOaOdau5ZMhaFDo0NquxJe")
    download_if_needed("nfm_dnn.pth", "1qSJ41pPwRo_F2IccmOvjwy4IZwYLqz1I")
    download_if_needed("nfm_user_embedding.pth", "1lZtXG6i7JX2dnCluuhJ_6avT8shsq719")
    download_if_needed("nfm_fm_layer.pth", "1OB6a_xq8Fa0Lto4NhOj_XVjLUnzEtqu7")
    download_if_needed("nfm_item_embedding.pth", "1WVOL-K2Quckjwg7ey8QnvHzj1gbWRGYZ")
    download_if_needed("neuralfm_model.pth", "1aQNmzeZEjOOKug8TqoUCq1kcIfyK7C2l")
    download_if_needed("neuralfm_embedding.pth", "1fnW0ptt-kA9BlWk1dXHtObaU7WUF1lUz")

def load_course_df():
    print("RAILWAY_ENVIRONMENT:", os.environ.get("RAILWAY_ENVIRONMENT"))
    file_id = os.environ.get("COURSE_CSV_DRIVE_ID")
    
    if os.environ.get("RAILWAY_ENVIRONMENT") and file_id and not os.path.exists("course_cleaned.csv"):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from: {url}")
        import gdown
        gdown.download(url, "course_cleaned.csv", quiet=False)
    
    return pd.read_csv(
        "course_cleaned.csv",
        usecols=[
            "course_id", "course_name", "image_url", "category",
            "calories", "sugar", "fiber", "ingredients", "cooking_directions"
        ]
    )

course_df = load_course_df()
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
]
CSV_COLUMNS = [
    "user_id", "name", "email", "password", "course_id", "rating", "course_name",
    "category", "image_url", "ingredients", "cooking_directions",
    "calories", "sugar", "fiber", "user_index"
]

from recommendation_logic import (
    get_model_recommendations,
    get_filtered_recommendations,
    classify_sugar_level,
    reinforcement_update_batch
)

# Load Data
file_path = "filtered_df.csv"
df = pd.read_csv(file_path)
course_df = load_course_df()
course_df["category"] = course_df["category"].str.lower()
scaler = joblib.load("trained_scaler.pkl")

# Neural FM Model
class NeuralFM(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, deep_layers):
        super(NeuralFM, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.fm_layer = nn.Linear(embed_dim * 2, 1)
        layers = []
        input_dim = embed_dim
        for layer_size in deep_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        fm_input = torch.cat([user_embed, item_embed], dim=1)
        interaction = user_embed * item_embed
        fm_output = self.fm_layer(fm_input)
        deep_output = self.dnn(interaction)
        return self.sigmoid(fm_output + deep_output).squeeze()

# Load model weights
num_users = 1575
num_items = 5754
embed_dim = 32
deep_layers = [64, 32]
model = NeuralFM(num_users, num_items, embed_dim, deep_layers)
model.fm_layer.load_state_dict(torch.load("nfm_fm_layer.pth"))
model.dnn.load_state_dict(torch.load("nfm_dnn.pth"))
model.user_embedding.load_state_dict(torch.load("nfm_user_embedding.pth"))
model.item_embedding.load_state_dict(torch.load("nfm_item_embedding.pth"))
model.eval()

# Thresholds for sugar levels
sugar_thresholds = {
    "high": {"max_sugar": 15, "max_calories": 350, "min_fiber": 2},
    "normal": {"max_sugar": 30, "max_calories": 500, "min_fiber": 1.5},
    "low": {"max_sugar": 40, "max_calories": 700, "min_fiber": 1}
}

layout = html.Div([
    dbc.Container([
        
        dcc.Location(id="logout-redirect", refresh=True),

        dbc.Row([
            dbc.Col(html.H4("Personalized Meal Recommendations", className="my-4")),
            dbc.Col(
                dbc.Button("Sign Out", id="logout-btn", color="danger", className="mt-4 float-end"),
                width="auto"
            )
        ]),
        dcc.Store(id="session-user-id", storage_type="session"),
        dcc.Store(id="session-username", storage_type="session"),
        dcc.Store(id="session-sugar", storage_type="session"),
        dcc.Store(id="session-bmi", storage_type="session"),  

        dcc.Store(id="expanded-cards", data=[], storage_type="memory"),
        html.Div(id="recommendation-section"),
        html.Div(id="feedback-msg", className="mt-3"),
        
        # Add modal component
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
                dbc.ModalBody(id="modal-content"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal", className="ms-auto")
                ),
            ],
            id="recipe-modal",
            size="lg",
            centered=True,
            is_open=False,
            className="modal-dialog-centered modal-dialog-scrollable recipe-modal-style"
        ),
        dbc.Toast(
        id="feedback-toast",
        header="Success",
        icon="success",
        is_open=False,
        children="", 
        duration=3000,
        dismissable=True,
        style={"position": "fixed", "top": 20, "right": 20, "width": 300, "zIndex": 9999}
    ),
    ], fluid=True, className="dashapp1-content")
], className="dashapp1-container")

# Display recommendations using session values
@dash.callback(
    Output("recommendation-section", "children"),
    Input("session-user-id", "data"),
    Input("session-sugar", "data"),
    Input("session-bmi", "data")  
)
def display_recommendations(user_id, sugar_value, bmi_value):
    
    if user_id is None or sugar_value is None:
        return dbc.Alert("You must be signed in to see recommendations.", color="danger")

    db = SessionLocal()
    try:
        result = db.execute(text("SELECT DISTINCT user_id, user_index FROM filtered_df WHERE user_index IS NOT NULL"))
        user_mapping = pd.DataFrame(result.fetchall(), columns=["user_id", "user_index"])
    finally:
        db.close()

    is_new_user = user_id not in user_mapping["user_id"].values

    if is_new_user:
        user_index = None
    else:
        user_index = user_mapping[user_mapping["user_id"] == user_id]["user_index"].values[0]

    num_items = len(course_df)

    filtered = get_filtered_recommendations(
    user_index=user_index,
    sugar_value=sugar_value,
    bmi_value=bmi_value,
    is_new_user=is_new_user,
    course_df=course_df,
    model=model,
    num_items=num_items,
    top_n=10
)

    db = SessionLocal()
    try:
        query = text("SELECT sugar_value, timestamp FROM sugar_levels WHERE user_id = :user_id ORDER BY timestamp ASC")
        result = db.execute(query, {"user_id": user_id}).fetchall()
        sugar_df = pd.DataFrame(result, columns=["sugar_value", "timestamp"])
    finally:
        db.close()

    print("Serving course_ids:", filtered["course_id"].tolist())

    sugar_chart = html.Div()
    if not sugar_df.empty:
        sugar_df["timestamp"] = pd.to_datetime(sugar_df["timestamp"])
        fig = px.line(
            sugar_df,
            x="timestamp",
            y="sugar_value",
            title="Your Blood Sugar Level Over Time",
            labels={"timestamp": "Date", "sugar_value": "Sugar Level (mg/dL)"},
            markers=True
        )
        fig.update_layout(
            margin={"t": 40, "b": 10},
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font_size=16,
            font=dict(size=10),
            xaxis=dict(
                title_font=dict(size=11),
                tickfont=dict(size=10),
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)'
            ),
            yaxis=dict(
                title_font=dict(size=11),
                tickfont=dict(size=10),
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)'
            )
        )

        sugar_chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    cards = []
    for _, row in filtered.iterrows():
        course_id = row["course_id"]
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    # Image 
                    dbc.CardImg(
                        src=row["image_url"], 
                        top=True,
                        className="course-card-img"
                    ),
                    # Title 
                    html.H6(
                        row["course_name"],  
                        className="card-title mb-3 course-card-title"
                    ),
                    # Descriptions
                    html.Div([
                        html.Span(f"{row['category'].title()}", className="mb-2"),
                    ], className="card-text"),
                    
                    html.Div([
                        # Left side - View More button
                        dbc.Button(
                            "View More",
                            id={"type": "view-more-btn", "index": int(course_id)},
                            color="link",
                            className="text-primary p-0 view-more-btn"
                        ),
                        # Right side - Like/Dislike buttons
                        dbc.ButtonGroup([
                            dbc.Button(
                                html.I(className="far fa-heart"), 
                                id={"type": "like-btn", "index": int(course_id)}, 
                                color="link",
                                className="text-dark size-sm like-btn"
                            ),
                            dbc.Button(
                                html.I(className="far fa-thumbs-down"),
                                id={"type": "dislike-btn", "index": int(course_id)}, 
                                color="link",
                                className="text-dark size-sm dislike-btn"
                            )
                        ])
                    ], className="d-flex justify-content-between align-items-center mb-2")
                ], className="p-4 course-card-body"),
            ], className="mb-4 h-100 card-hover course-card-body")
        )
    return html.Div([
        html.Div(
            id="carousel-wrapper",
            children=[
                html.Div(
                    card,
                    className="carousel-card"
                ) for card in cards
            ],

            className="carousel-container"
        ),

        # Scroll buttons
        html.Div([
            dbc.Button(
                html.I(className="fas fa-chevron-left"),
                id="scroll-left",
                color="light",
                className="carousel-button rounded-circle me-2",
            ),
            dbc.Button(
                html.I(className="fas fa-chevron-right"),
                id="scroll-right",
                color="light",
                className="carousel-button rounded-circle",
            )
        ], className="carousel-button-wrapper"),

        html.Div([
            html.Hr(className="mt-5"),  
            sugar_chart
        ], className="mt-5 mb-4"), 

        dcc.Store(id="carousel-scroll")
    ])

@dash.callback(
    Output("feedback-msg", "children", allow_duplicate=True),
    Output("feedback-toast", "is_open", allow_duplicate=True), 
    Output("feedback-toast", "children", allow_duplicate=True),
    Input({"type": "like-btn", "index": ALL}, "n_clicks"),
    Input({"type": "dislike-btn", "index": ALL}, "n_clicks"),
    State({"type": "like-btn", "index": ALL}, "n_clicks_timestamp"),
    State({"type": "dislike-btn", "index": ALL}, "n_clicks_timestamp"),
    State("session-user-id", "data"),
    prevent_initial_call="initial_duplicate"
)
def handle_rating(likes, dislikes, like_times, dislike_times, user_id):
    if not user_id:
        raise dash.exceptions.PreventUpdate

    latest_like = max([(t or 0, i) for i, t in enumerate(like_times or [])], default=(0, None))
    latest_dislike = max([(t or 0, i) for i, t in enumerate(dislike_times or [])], default=(0, None))

    if latest_like[0] == 0 and latest_dislike[0] == 0:
        raise dash.exceptions.PreventUpdate

    course_id = (
        ctx.inputs_list[0][latest_like[1]]["id"]["index"]
        if latest_like[0] > latest_dislike[0]
        else ctx.inputs_list[1][latest_dislike[1]]["id"]["index"]
    )
    rating = 1 if latest_like[0] > latest_dislike[0] else 0

    try:
        course_row = course_df[course_df['course_id'] == course_id].iloc[0]

        db = SessionLocal()
        try:
            result = db.execute(text("SELECT DISTINCT user_id, user_index FROM filtered_df WHERE user_index IS NOT NULL"))
            user_mapping = pd.DataFrame(result.fetchall(), columns=["user_id", "user_index"])
        finally:
            db.close()


        if user_id in user_mapping["user_id"].values:
            user_index = user_mapping[user_mapping["user_id"] == user_id]["user_index"].values[0]
        else:
            df_updated = pd.read_csv(file_path)
            user_index = df_updated["user_index"].max() + 1 if not df_updated.empty else 0

        course_index = int(course_df[course_df["course_id"] == course_id].index[0])

        # Fetch user name
        db = SessionLocal()
        try:
            query = text("SELECT name FROM users WHERE user_id = :user_id")
            result = db.execute(query, {"user_id": user_id}).fetchone()
            user_name = result[0] if result else "Unknown"
        finally:
            db.close()
        # Normalized nutrition values
        scaled_vals = scaler.transform([[course_row["calories"], course_row["sugar"], course_row["fiber"]]])
        calories_norm, sugar_norm, fiber_norm = scaled_vals[0]

        category_map = {"appetizer": 0, "main-dish": 1, "dessert": 2}
        category_code = category_map.get(course_row["category"].lower(), -1)
    
        new_row_csv = pd.DataFrame([{
            "user_id": user_id,
            "name": user_name,
            "email": "", 
            "password": "", 
            "course_id": course_id,
            "rating": rating,
            "course_name": course_row["course_name"],
            "category": category_code,
            "image_url": course_row["image_url"],
            "ingredients": course_row["ingredients"],
            "cooking_directions": course_row["cooking_directions"],
            "calories": calories_norm,
            "sugar": sugar_norm,
            "fiber": fiber_norm,
            "user_index": user_index
        }])[CSV_COLUMNS]

        # Trimmed row for DB (no email/password)
        new_row_db = new_row_csv.drop(columns=["email", "password"])

        new_row_csv.to_csv(file_path, mode='a', header=False, index=False)
    
        db = SessionLocal()
        try:
            for _, row in new_row_db.iterrows():
                insert_filtered = text("""
                    INSERT INTO filtered_df (
                        user_id, name, course_id, rating, course_name,
                        category, image_url, ingredients, cooking_directions,
                        calories, sugar, fiber, user_index
                    ) VALUES (
                        :user_id, :name, :course_id, :rating, :course_name,
                        :category, :image_url, :ingredients, :cooking_directions,
                        :calories, :sugar, :fiber, :user_index
                    )
                """)
                db.execute(insert_filtered, row.to_dict())

            insert_rl = text("""
                INSERT INTO rl_buffer (user_index, course_index, rating)
                VALUES (:user_index, :course_index, :rating)
            """)
     
            db.execute(insert_rl, {
                "user_index": int(user_index),
                "course_index": int(course_index),
                "rating": int(rating)
            })
            print(" handle_rating triggered - user_id:", user_id)


            db.commit()
            count_result = db.execute(text("SELECT COUNT(*) FROM rl_buffer"))
            count = count_result.scalar()
        finally:
            db.close()

        print(f"Current rl_buffer count = {count}")
        if count >= 6:
            print(" Reinforcement condition met. Proceeding...")
            db = SessionLocal()  # New session
            try:
                df_buffer = pd.read_sql("SELECT * FROM rl_buffer", db.connection())

                user_tensor = torch.tensor(df_buffer["user_index"].astype(int).values, dtype=torch.long)
                item_tensor = torch.tensor(df_buffer["course_index"].astype(int).values, dtype=torch.long)
                label_tensor = torch.tensor(df_buffer["rating"].astype(float).values, dtype=torch.float32)

                # Check recommendations BEFORE training
                sample_user_index = int(user_tensor[0].item())
                recs_before = get_model_recommendations(sample_user_index, model, course_df, num_items)
                print("Before RL update:", recs_before["course_id"].tolist()[:5])

                # Apply reinforcement update
                reinforcement_update_batch(model, user_tensor, item_tensor, label_tensor)

                # Check recommendations AFTER training
                recs_after = get_model_recommendations(sample_user_index, model, course_df, num_items)
                print("After  RL update:", recs_after["course_id"].tolist()[:5])
                
                db.execute(text("DELETE FROM rl_buffer"))
                db.commit()
            finally:
                db.close()

            
        return f" Feedback recorded for course {course_id}", True, f" Feedback saved for {course_row['course_name']}"

    except Exception as e:
        return f" Error saving feedback: {e}", True, f" Error: {e}"


@dash.callback(
    [Output("recipe-modal", "is_open"),
     Output("modal-title", "children"),
     Output("modal-content", "children")],
    [Input({"type": "view-more-btn", "index": ALL}, "n_clicks"),
     Input("close-modal", "n_clicks")],
    [State("recipe-modal", "is_open")],
    prevent_initial_call=True
)


def toggle_modal(view_clicks, close_clicks, is_open):
    if ctx.triggered_id == "close-modal":
        return False, "", None
    
    if not any(view_clicks):
        raise dash.exceptions.PreventUpdate
    
    triggered_id = ctx.triggered_id
    if triggered_id:
        course_id = triggered_id["index"]
        course = course_df[course_df["course_id"] == course_id].iloc[0]
        
        modal_content = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(
                        src=course["image_url"],
                        className="modal-course-img"
                    ),
                    html.Div([
                        html.H5("Details", className="mb-3 modal-section-title"),  
                        dbc.Row([
                            # Left column
                            dbc.Col([
                                html.P(f"Category: {course['category'].title()}", className="mb-2"),
                                html.P(f"Calories: {course['calories']:.2f}", className="mb-2"),
                            ], width=6),
                            # Right column
                            dbc.Col([
                                html.P(f"Sugar: {course['sugar']:.2f}g", className="mb-2"),
                                html.P(f"Fiber: {course['fiber']:.2f}g", className="mb-2"),
                            ], width=6),
                        ]),
                    ], className="mb-4"),
                ], width=12, lg=6),
                dbc.Col([
                    html.Div([
                        html.H5("Ingredients", className="mb-3"),
                        html.P(course["ingredients"],  className="mb-4 pre-line-text"),
                        html.H5("Recipe", className="mb-3"),
                        html.P(course["cooking_directions"], 
                              className="pre-line-text")
                    ])
                ], width=12, lg=6)
            ])
        ], fluid=True, className="px-0")
        
        return True, course["course_name"], modal_content
    
    return False, "", None

@dash.callback(
    Output("logout-redirect", "pathname"),
    Output("session-user-id", "data"),
    Output("session-username", "data"),
    Output("session-sugar", "data"),
    Output("session-bmi", "data"),
    Input("logout-btn", "n_clicks"),
    prevent_initial_call=True
)
def logout_user(n_clicks):
    return "/login", None, None, None, None

def toggle_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

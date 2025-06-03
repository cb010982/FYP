import dash
from dash import dcc, html, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import torch
import torch.nn as nn
import sqlite3

# Load data
file_path = "filtered_df.csv"
df = pd.read_csv(file_path)
course_df = pd.read_csv("course_cleaned.csv")
course_df["category"] = course_df["category"].str.lower()

if "user_index" not in df.columns:
    unique_users = df["user_id"].unique()
    user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
    df["user_index"] = df["user_id"].map(user_mapping)

# Model
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

# Model init
num_users = 1575
num_items = 5751
embed_dim = 32
deep_layers = [128, 64]
model = NeuralFM(num_users, num_items, embed_dim, deep_layers)
model.fm_layer.load_state_dict(torch.load("nfm_fm_layer.pth"))
model.dnn.load_state_dict(torch.load("nfm_dnn.pth"))
model.user_embedding.load_state_dict(torch.load("nfm_user_embedding.pth"))
model.item_embedding.load_state_dict(torch.load("nfm_item_embedding.pth"))
model.eval()

# Sugar thresholds
sugar_thresholds = {
    "high": {"max_sugar": 15, "max_calories": 350, "min_fiber": 2},
    "normal": {"max_sugar": 30, "max_calories": 500, "min_fiber": 1.5},
    "low": {"max_sugar": 40, "max_calories": 700, "min_fiber": 1}
}

def classify_sugar_level(sugar_value):
    if sugar_value > 180:
        return "high"
    elif 100 <= sugar_value <= 180:
        return "normal"
    else:
        return "low"

layout = dbc.Container([
    html.H2("\U0001F37D\ufe0f Personalized Meal Recommendations", className="my-4 text-center"),
    dbc.Row([
        dbc.Col([
            dcc.RadioItems(options=["Sign In", "Sign Up"], value="Sign In", id="auth-choice", inline=True),
            dcc.Input(id="username", placeholder="Enter username", type="text", className="form-control my-2"),
            dbc.Button("Submit", id="submit-auth", className="mb-3", color="primary"),
            html.Div(id="auth-msg"),
            dcc.Input(id="sugar", type="number", placeholder="Enter sugar level (mg/dL)", min=80, max=300, step=0.1, className="form-control mb-3"),
        ], width=4),
        dbc.Col([html.Div(id="recommendation-section")], width=8)
    ]),
    dcc.Store(id="session-user-id"),
    dcc.Store(id="session-sugar")
], fluid=True)

@dash.callback(
    Output("auth-msg", "children"),
    Output("session-user-id", "data"),
    Output("session-sugar", "data"),
    Input("submit-auth", "n_clicks"),
    State("auth-choice", "value"),
    State("username", "value"),
    State("sugar", "value")
)
def handle_auth(n, choice, username, sugar):
    if not n or not username:
        return "", None, None

    global df
    if choice == "Sign In":
        if username in df["Name"].values:
            user_id = df[df["Name"] == username]["user_index"].values[0]
            return f"Welcome back, {username}! Your User ID is {user_id}.", user_id, sugar
        else:
            return "User not found. Please sign up.", None, None
    else:
        if username in df["Name"].values:
            return "Username already exists. Try signing in.", None, None
        else:
            new_user_index = df["user_index"].max() + 1 if len(df) > 0 else 0
            new_user = pd.DataFrame({
                "user_id": [new_user_index],
                "user_index": [new_user_index],
                "Name": [username]
            })
            df = pd.concat([df, new_user], ignore_index=True)
            df.to_csv(file_path, index=False)

            return (
                f"User {username} created! Note: You won't receive personalized recommendations until the model is retrained.",
                new_user_index,
                sugar
            )


@dash.callback(
    Output("recommendation-section", "children"),
    Input("session-user-id", "data"),
    Input("session-sugar", "data")
)
def recommend_meals(user_id, sugar_value):
    if user_id is None or sugar_value is None:
        return dbc.Alert("Please sign in to view recommendations.", color="warning")

    def get_model_recommendations(user_id, top_n=50):
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        item_tensor = torch.arange(num_items, dtype=torch.long)
        with torch.no_grad():
            scores = model(user_tensor.repeat(num_items), item_tensor).squeeze()
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        return course_df.iloc[sorted_indices[:top_n]]

    def get_filtered_recommendations(user_id, sugar_value, top_n=10):
        sugar_category = classify_sugar_level(sugar_value)
        thresholds = sugar_thresholds[sugar_category]
        recs = get_model_recommendations(user_id)
        filtered = recs[
            (recs["sugar"] <= thresholds["max_sugar"]) &
            (recs["calories"] <= thresholds["max_calories"]) &
            (recs["fiber"] >= thresholds["min_fiber"])
        ]
        return filtered.head(top_n)

    filtered = get_filtered_recommendations(user_id, sugar_value)
    if filtered.empty:
        return html.P("No meals match your health filters.")

    cards = []
    for _, row in filtered.iterrows():
        course_id = row["course_id"]
        cards.append(
            dbc.Card([
                dbc.CardImg(src=row["image_url"], top=True, style={"width": "200px"}),
                dbc.CardBody([
                    html.H5(row["course_name"]),
                    html.P(f"\U0001F525 Calories: {row['calories']:.2f}"),
                    html.P(f"\U0001F36C Sugar: {row['sugar']:.2f}g"),
                    html.P(f"\U0001F33F Fiber: {row['fiber']:.2f}g"),
                    html.P(f"\U0001F955 Ingredients: {row['ingredients'][:500]}..."),
                    html.P(f"\U0001F4D6 Recipe: {row['cooking_directions'][:500]}..."),
                    dbc.ButtonGroup([
                        dbc.Button("👍", id={"type": "like-btn", "index": int(course_id)}, color="success", size="sm"),
                        dbc.Button("👎", id={"type": "dislike-btn", "index": int(course_id)}, color="danger", size="sm")
                    ], size="sm", className="mt-2")
                ])
            ], className="mb-4")
        )

    return dbc.Row([dbc.Col(card, width=6) for card in cards])

@dash.callback(
    Output("auth-msg", "children", allow_duplicate=True),
    Input({"type": "like-btn", "index": ALL}, "n_clicks"),
    Input({"type": "dislike-btn", "index": ALL}, "n_clicks"),
    State("session-user-id", "data"),
    prevent_initial_call="initial_duplicate"
)
def handle_rating_append(likes, dislikes, user_id):
    triggered = ctx.triggered_id
    if not triggered or user_id is None:
        raise dash.exceptions.PreventUpdate

    course_id = triggered['index']
    rating = 1 if triggered['type'] == 'like-btn' else 0

    # Load course details from course_df
    try:
        course_row = course_df[course_df['course_id'] == course_id].iloc[0]
        user_name = df[df["user_index"] == user_id]["Name"].values[0]

        new_row = pd.DataFrame([{
            "user_id": user_id,
            "Name": user_name,
            "course_id": course_id,
            "rating": rating,
            "course_name": course_row["course_name"],
            "category": course_row["category"],
            "image_url": course_row["image_url"],
            "ingredients": course_row["ingredients"],
            "cooking_directions": course_row["cooking_directions"],
            "calories": course_row["calories"],
            "sugar": course_row["sugar"],
            "fiber": course_row["fiber"]
        }])

        # Append to filtered_df.csv and table
        new_row.to_csv(file_path, mode='a', header=False, index=False)
        conn = sqlite3.connect("meals.db")
        new_row.to_sql("filtered_df", conn, if_exists="append", index=False)
        conn.close()

        return f"✅ Feedback recorded for course {course_id}"

    except Exception as e:
        return f"❌ Failed to record rating: {e}"
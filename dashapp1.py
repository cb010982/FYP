


# import dash
# from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH
# import dash_bootstrap_components as dbc
# import pandas as pd
# import torch
# import torch.nn as nn
# import sqlite3
# import joblib
# from sklearn.preprocessing import MinMaxScaler
# import plotly.express as px
# from db import SessionLocal
# from sqlalchemy import text

# external_stylesheets = [
#     dbc.themes.BOOTSTRAP,
#     'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
# ]
# CSV_COLUMNS = [
#     "user_id", "name", "email", "password", "course_id", "rating", "course_name",
#     "category", "image_url", "ingredients", "cooking_directions",
#     "calories", "sugar", "fiber", "user_index"
# ]
# # Load Data
# file_path = "filtered_df.csv"
# df = pd.read_csv(file_path)
# course_df = pd.read_csv("course_cleaned.csv")
# course_df["category"] = course_df["category"].str.lower()
# scaler = joblib.load("trained_scaler.pkl")

# if "user_index" not in df.columns:
#     unique_users = df["user_id"].unique()
#     user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
#     df["user_index"] = df["user_id"].map(user_mapping)

# # Neural FM Model
# class NeuralFM(nn.Module):
#     def __init__(self, num_users, num_items, embed_dim, deep_layers):
#         super(NeuralFM, self).__init__()
#         self.user_embedding = nn.Embedding(num_users, embed_dim)
#         self.item_embedding = nn.Embedding(num_items, embed_dim)
#         self.fm_layer = nn.Linear(embed_dim * 2, 1)
#         layers = []
#         input_dim = embed_dim
#         for layer_size in deep_layers:
#             layers.append(nn.Linear(input_dim, layer_size))
#             layers.append(nn.ReLU())
#             input_dim = layer_size
#         layers.append(nn.Linear(input_dim, 1))
#         self.dnn = nn.Sequential(*layers)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, user, item):
#         user_embed = self.user_embedding(user)
#         item_embed = self.item_embedding(item)
#         fm_input = torch.cat([user_embed, item_embed], dim=1)
#         interaction = user_embed * item_embed
#         fm_output = self.fm_layer(fm_input)
#         deep_output = self.dnn(interaction)
#         return self.sigmoid(fm_output + deep_output).squeeze()

# # Load model weights
# num_users = 1575
# num_items = 5754
# embed_dim = 32
# deep_layers = [64, 32]
# model = NeuralFM(num_users, num_items, embed_dim, deep_layers)
# model.fm_layer.load_state_dict(torch.load("nfm_fm_layer.pth"))
# model.dnn.load_state_dict(torch.load("nfm_dnn.pth"))
# model.user_embedding.load_state_dict(torch.load("nfm_user_embedding.pth"))
# model.item_embedding.load_state_dict(torch.load("nfm_item_embedding.pth"))
# model.eval()

# # Thresholds for sugar levels
# sugar_thresholds = {
#     "high": {"max_sugar": 15, "max_calories": 350, "min_fiber": 2},
#     "normal": {"max_sugar": 30, "max_calories": 500, "min_fiber": 1.5},
#     "low": {"max_sugar": 40, "max_calories": 700, "min_fiber": 1}
# }

# def classify_sugar_level(sugar_value):
#     sugar_value = float(sugar_value)
#     if sugar_value > 180:
#         return "high"
#     elif 100 <= sugar_value <= 180:
#         return "normal"
#     else:
#         return "low"

# # Dash layout (no login/signup here anymore!)
# layout = html.Div([
#     dbc.Container([
#         html.H4("Personalized Meal Recommendations", className="my-4 text-center"),
#         dcc.Store(id="session-user-id", storage_type="session"),
#         dcc.Store(id="session-username", storage_type="session"),
#         dcc.Store(id="session-sugar", storage_type="session"),
#         dcc.Store(id="session-bmi", storage_type="session"),  # BMI

#         dcc.Store(id="expanded-cards", data=[], storage_type="memory"),
#         html.Div(id="recommendation-section"),
#         html.Div(id="feedback-msg", className="mt-3"),
#         # Add modal component
#         dbc.Modal(
#             [
#                 dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
#                 dbc.ModalBody(id="modal-content"),
#                 dbc.ModalFooter(
#                     dbc.Button("Close", id="close-modal", className="ms-auto")
#                 ),
#             ],
#             id="recipe-modal",
#             size="lg",
#             centered=True,
#             is_open=False,
#             style={
#                 "display": "flex",
#                 "alignItems": "center",
#                 "justifyContent": "center",
#             },
#             className="modal-dialog-centered modal-dialog-scrollable"
#         ),
#     ], fluid=True, className="dashapp1-content")
# ], className="dashapp1-container")

# # Display recommendations using session values
# @dash.callback(
#     Output("recommendation-section", "children"),
#     Input("session-user-id", "data"),
#     Input("session-sugar", "data"),
#     Input("session-bmi", "data")  #BMI added
# )
# def display_recommendations(user_id, sugar_value, bmi_value):#bmi added
    
#     if user_id is None or sugar_value is None:
#         return dbc.Alert("You must be signed in to see recommendations.", color="danger")

#     # user_mapping = df[["user_id", "user_index"]].drop_duplicates()
#     db = SessionLocal()
#     try:
#         result = db.execute(text("SELECT DISTINCT user_id, user_index FROM filtered_df WHERE user_index IS NOT NULL"))
#         user_mapping = pd.DataFrame(result.fetchall(), columns=["user_id", "user_index"])
#     finally:
#         db.close()

#     is_new_user = user_id not in user_mapping["user_id"].values

#     if is_new_user:
#         user_index = None
#     else:
#         user_index = user_mapping[user_mapping["user_id"] == user_id]["user_index"].values[0]

#     def get_model_recommendations(user_index, top_n=50):
#         user_tensor = torch.tensor([user_index], dtype=torch.long)
#         item_tensor = torch.arange(num_items, dtype=torch.long)
#         with torch.no_grad():
#             scores = model(user_tensor.repeat(num_items), item_tensor).squeeze()
#         sorted_indices = torch.argsort(scores, descending=True).tolist()
#         return course_df.iloc[sorted_indices[:top_n]]

#     def get_filtered_recommendations(user_index, sugar_value, bmi_value, top_n=10):
#         sugar_category = classify_sugar_level(sugar_value)
#         thresholds = sugar_thresholds[sugar_category]
#         recs = course_df.copy() if is_new_user else get_model_recommendations(user_index)
#         if is_new_user and bmi_value is not None: #added new
#             bmi_value = float(bmi_value)
#             if bmi_value >= 30: #added new
#                 thresholds["max_calories"] -= 100
#                 thresholds["max_sugar"] -= 5
#                 thresholds["min_fiber"] += 0.5
#         filtered = recs[
#             (recs["sugar"] <= thresholds["max_sugar"]) &
#             (recs["calories"] <= thresholds["max_calories"]) &
#             (recs["fiber"] >= thresholds["min_fiber"])
#         ]
#         return filtered.head(top_n)

#     filtered = get_filtered_recommendations(user_index, sugar_value, bmi_value)

#     if (filtered.empty):
#         return html.P("No meals match your health preferences.")
#     # Load sugar level history from DB
#     db = SessionLocal()
#     try:
#         query = text("SELECT sugar_value, timestamp FROM sugar_levels WHERE user_id = :user_id ORDER BY timestamp ASC")
#         result = db.execute(query, {"user_id": user_id}).fetchall()
#         sugar_df = pd.DataFrame(result, columns=["sugar_value", "timestamp"])
#     finally:
#         db.close()


#     sugar_chart = html.Div()
#     if not sugar_df.empty:
#         sugar_df["timestamp"] = pd.to_datetime(sugar_df["timestamp"])
#         fig = px.line(
#             sugar_df,
#             x="timestamp",
#             y="sugar_value",
#             title="Your Blood Sugar Level Over Time",
#             labels={"timestamp": "Date", "sugar_value": "Sugar Level (mg/dL)"},
#             markers=True
#         )
#         fig.update_layout(
#             margin={"t": 40, "b": 10},
#             height=300,
#             plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
#             paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
#             title_font_size=18,
#             xaxis=dict(
#                 showgrid=True,
#                 gridcolor='rgba(211, 211, 211, 0.5)'  # Light grey grid with 50% opacity
#             ),
#             yaxis=dict(
#                 showgrid=True,
#                 gridcolor='rgba(211, 211, 211, 0.5)'  # Light grey grid with 50% opacity
#             )
#         )
#         sugar_chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

#     cards = []
#     for _, row in filtered.iterrows():
#         course_id = row["course_id"]
#         cards.append(
#             dbc.Card([
#                 dbc.CardBody([
#                     # Image first
#                     dbc.CardImg(
#                         src=row["image_url"], 
#                         top=True,
#                         style={
#                             "width": "100%",
#                             "height": "200px",
#                             "object-fit": "cover",
#                             "borderRadius": "10px"  # Added border radius
#                         }
#                     ),
#                     # Title second
#                     html.H6(
#                         row["course_name"], 
#                         className="card-title mb-3",  # Removed text-center class
#                         style={
#                             "padding-top":"10px",
#                             "textAlign": "left"  # Explicitly set left alignment
#                         }
#                     ),
#                     # Descriptions third
#                     html.Div([
#                         html.Span(f"{row['category'].title()}", className="mb-2"),
#                         # html.P(f" Calories: {row['calories']:.2f}", className="mb-2"),
#                         # html.P(f" Sugar: {row['sugar']:.2f}g", className="mb-2"),
#                         # html.P(f" Fiber: {row['fiber']:.2f}g", className="mb-2"),
#                         # html.P(f" Ingredients: {row['ingredients'][:100]}...", className="mb-2"),  # Shortened preview
#                         # html.P(f" Recipe: {row['cooking_directions'][:100]}...", className="mb-2"),  # Shortened preview
#                     ], className="card-text"),

#                     # Replace the View More button and ButtonGroup section with:
#                     html.Div([
#                         # Left side - View More button
#                         dbc.Button(
#                             "View More",
#                             id={"type": "view-more-btn", "index": int(course_id)},
#                             color="link",
#                             className="text-primary p-0",
#                             style={
#                                 "textDecoration": "none",
#                                 "fontSize": "14px",
#                                 "fontWeight": "500"
#                             }
#                         ),
#                         # Right side - Like/Dislike buttons
#                         dbc.ButtonGroup([
#                             dbc.Button(
#                                 html.I(className="far fa-heart"), 
#                                 id={"type": "like-btn", "index": int(course_id)}, 
#                                 color="link",
#                                 className="text-dark",
#                                 size="sm",
#                                 style={
#                                     "border": "none",
#                                     "background": "none",
#                                     "padding": "0px",
#                                     "marginRight": "10px",
#                                     "fontSize": "20px"
#                                 }
#                             ),
#                             dbc.Button(
#                                 html.I(className="far fa-thumbs-down"),
#                                 id={"type": "dislike-btn", "index": int(course_id)}, 
#                                 color="link",
#                                 className="text-dark",
#                                 size="sm",
#                                 style={
#                                     "border": "none",
#                                     "background": "none",
#                                     "padding": "0px",
#                                     "fontSize": "20px"
#                                 }
#                             )
#                         ])
#                     ], className="d-flex justify-content-between align-items-center mb-2")
#                 ], className="p-4", 
#                 style={
#                     "backgroundColor": "#faf7f2",  # Light beige color
#                     "border": "1px solid #e9ecef",
#                     "borderRadius": "15px",
#                     "boxShadow": "none"
#                 })
#             ], className="mb-4 h-100 card-hover",
#             style={
#                 "backgroundColor": "#faf7f2",
#                 "border": "1px solid #e9ecef",
#                 "borderRadius": "15px",
#                 "boxShadow": "none"
#             })
#         )
#     return html.Div([
#         # Carousel of recommendations first
#         html.Div(
#             id="carousel-wrapper",
#             children=[
#                 html.Div(
#                     card,
#                     style={
#                         "minWidth": "300px",
#                         "maxWidth": "300px",
#                         "marginRight": "15px",
#                         "flex": "0 0 auto",
#                         "scrollSnapAlign": "start"
#                     }
#                 ) for card in cards
#             ],
#             style={
#                 "display": "flex",
#                 "overflowX": "auto",
#                 "scrollBehavior": "smooth",
#                 "padding": "10px",
#                 "gap": "10px",
#                 "scrollSnapType": "x mandatory",
#                 "msOverflowStyle": "none",
#                 "scrollbarWidth": "none",
#                 "::-webkit-scrollbar": {
#                     "display": "none"
#                 }
#             },
#             className="carousel-container"
#         ),

#         # Scroll buttons
#         html.Div([
#             dbc.Button(
#                 html.I(className="fas fa-chevron-left"),
#                 id="scroll-left",
#                 color="light",
#                 className="me-2 rounded-circle",
#                 style={
#                     "width": "40px",
#                     "height": "40px",
#                     "padding": "0",
#                     "display": "flex",
#                     "alignItems": "center",
#                     "justifyContent": "center",
#                     "backgroundColor": "#ffffff",
#                     "border": "1px solid #dee2e6",
#                     "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
#                 }
#             ),
#             dbc.Button(
#                 html.I(className="fas fa-chevron-right"),
#                 id="scroll-right",
#                 color="light",
#                 className="rounded-circle",
#                 style={
#                     "width": "40px",
#                     "height": "40px",
#                     "padding": "0",
#                     "display": "flex",
#                     "alignItems": "center",
#                     "justifyContent": "center",
#                     "backgroundColor": "#ffffff",
#                     "border": "1px solid #dee2e6",
#                     "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
#                 }
#             )
#         ], className="d-flex mt-3", style={"position": "fixed", "right": "80px"}),

#         # Sugar chart moved to bottom with padding
#         html.Div([
#             html.Hr(className="mt-5"),  # Add a divider
#             sugar_chart
#         ], className="mt-5 mb-4"),  # Add top and bottom margin

#         dcc.Store(id="carousel-scroll")
#     ])
# def reinforcement_update_batch(user_tensor, item_tensor, label_tensor):
#     print(f"🧠 Starting batch training on {len(user_tensor)} samples...")
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.BCELoss()

#     optimizer.zero_grad()
#     predictions = model(user_tensor, item_tensor)
#     loss = criterion(predictions, label_tensor)
#     loss.backward()
#     optimizer.step()

#     model.eval()
#     print("✅ Reinforcement batch update done")

# # Handle likes and dislikes
# @dash.callback(
#     Output("feedback-msg", "children", allow_duplicate=True),
#     Input({"type": "like-btn", "index": ALL}, "n_clicks"),
#     Input({"type": "dislike-btn", "index": ALL}, "n_clicks"),
#     State({"type": "like-btn", "index": ALL}, "n_clicks_timestamp"),
#     State({"type": "dislike-btn", "index": ALL}, "n_clicks_timestamp"),
#     State("session-user-id", "data"),
#     prevent_initial_call="initial_duplicate"
# )
# def handle_rating(likes, dislikes, like_times, dislike_times, user_id):
#     if not user_id:
#         raise dash.exceptions.PreventUpdate

#     latest_like = max([(t or 0, i) for i, t in enumerate(like_times or [])], default=(0, None))
#     latest_dislike = max([(t or 0, i) for i, t in enumerate(dislike_times or [])], default=(0, None))

#     if latest_like[0] == 0 and latest_dislike[0] == 0:
#         raise dash.exceptions.PreventUpdate

#     course_id = (
#         ctx.inputs_list[0][latest_like[1]]["id"]["index"]
#         if latest_like[0] > latest_dislike[0]
#         else ctx.inputs_list[1][latest_dislike[1]]["id"]["index"]
#     )
#     rating = 1 if latest_like[0] > latest_dislike[0] else 0

#     try:
#         course_row = course_df[course_df['course_id'] == course_id].iloc[0]

#         # 🧠 Compute NeuralFM-compatible indices (safe for new users)
#         # user_mapping = df[["user_id", "user_index"]].drop_duplicates()
#         db = SessionLocal()
#         try:
#             result = db.execute(text("SELECT DISTINCT user_id, user_index FROM filtered_df WHERE user_index IS NOT NULL"))
#             user_mapping = pd.DataFrame(result.fetchall(), columns=["user_id", "user_index"])
#         finally:
#             db.close()


#         if user_id in user_mapping["user_id"].values:
#             user_index = user_mapping[user_mapping["user_id"] == user_id]["user_index"].values[0]
#         else:
#             # user_index = df["user_index"].max() + 1 if not df.empty else 0
#             df_updated = pd.read_csv(file_path)
#             user_index = df_updated["user_index"].max() + 1 if not df_updated.empty else 0

#         course_index = int(course_df[course_df["course_id"] == course_id].index[0])

#         # Fetch user name
#         db = SessionLocal()
#         try:
#             query = text("SELECT name FROM users WHERE user_id = :user_id")
#             result = db.execute(query, {"user_id": user_id}).fetchone()
#             user_name = result[0] if result else "Unknown"
#         finally:
#             db.close()
#         # Normalized nutrition values
#         scaled_vals = scaler.transform([[course_row["calories"], course_row["sugar"], course_row["fiber"]]])
#         calories_norm, sugar_norm, fiber_norm = scaled_vals[0]

#         category_map = {"appetizer": 0, "main-dish": 1, "dessert": 2}
#         category_code = category_map.get(course_row["category"].lower(), -1)

#         # new_row = pd.DataFrame([{
#         #     "user_id": user_id,
#         #     "name": user_name,
#         #     "course_id": course_id,
#         #     "user_index": user_index,  # ✅ NOW this is defined correctly
#         #     "rating": rating,
#         #     "course_name": course_row["course_name"],
#         #     "category": category_code,
#         #     "image_url": course_row["image_url"],
#         #     "ingredients": course_row["ingredients"],
#         #     "cooking_directions": course_row["cooking_directions"],
#         #     "calories": calories_norm,
#         #     "sugar": sugar_norm,
#         #     "fiber": fiber_norm
#         # }])
#         # new_row["user_index"] = user_index

#         # 🧠 Compute NeuralFM-compatible indices
#         # user_mapping = df[["user_id", "user_index"]].drop_duplicates()
#         # user_index = user_mapping[user_mapping["user_id"] == user_id]["user_index"].values[0]
#         # 🧠 Compute NeuralFM-compatible indices (safe for new users)
    
#         new_row_csv = pd.DataFrame([{
#             "user_id": user_id,
#             "name": user_name,
#             "email": "",  # or actual
#             "password": "",  # or actual
#             "course_id": course_id,
#             "rating": rating,
#             "course_name": course_row["course_name"],
#             "category": category_code,
#             "image_url": course_row["image_url"],
#             "ingredients": course_row["ingredients"],
#             "cooking_directions": course_row["cooking_directions"],
#             "calories": calories_norm,
#             "sugar": sugar_norm,
#             "fiber": fiber_norm,
#             "user_index": user_index
#         }])[CSV_COLUMNS]

#         # Trimmed row for DB (no email/password)
#         new_row_db = new_row_csv.drop(columns=["email", "password"])
#         # # ✅ Insert into RL buffer
#         # # ✅ Save to CSV and DB
#         # new_row.to_csv(file_path, mode='a', header=False, index=False)
#         # conn = sqlite3.connect("meals.db")
#         # cursor = conn.cursor()
#         # new_row.to_sql("filtered_df", conn, if_exists="append", index=False)
#         # ✅ Save to CSV (includes email, password)
#         new_row_csv.to_csv(file_path, mode='a', header=False, index=False)
#         # ✅ Save to SQLite (excludes email, password)
#         db = SessionLocal()
#         try:
#             for _, row in new_row_db.iterrows():
#                 insert_filtered = text("""
#                     INSERT INTO filtered_df (
#                         user_id, name, course_id, rating, course_name,
#                         category, image_url, ingredients, cooking_directions,
#                         calories, sugar, fiber, user_index
#                     ) VALUES (
#                         :user_id, :name, :course_id, :rating, :course_name,
#                         :category, :image_url, :ingredients, :cooking_directions,
#                         :calories, :sugar, :fiber, :user_index
#                     )
#                 """)
#                 db.execute(insert_filtered, row.to_dict())

#             insert_rl = text("""
#                 INSERT INTO rl_buffer (user_index, course_index, rating)
#                 VALUES (:user_index, :course_index, :rating)
#             """)
#             # db.execute(insert_rl, {
#             #     "user_index": user_index,
#             #     "course_index": course_index,
#             #     "rating": rating
#             # })
#             db.execute(insert_rl, {
#                 "user_index": int(user_index),
#                 "course_index": int(course_index),
#                 "rating": int(rating)
#             })
#             print("✅ handle_rating triggered - user_id:", user_id)


#             db.commit()
#             count_result = db.execute(text("SELECT COUNT(*) FROM rl_buffer"))
#             count = count_result.scalar()
#         finally:
#             db.close()

#         # ✅ Only enter this block if needed
#         if count >= 50:
#             db = SessionLocal()  # New session
#             try:
#                 df_buffer = pd.read_sql("SELECT * FROM rl_buffer", db.connection())

#                 user_tensor = torch.tensor(df_buffer["user_index"].astype(int).values, dtype=torch.long)
#                 item_tensor = torch.tensor(df_buffer["course_index"].astype(int).values, dtype=torch.long)
#                 label_tensor = torch.tensor(df_buffer["rating"].astype(float).values, dtype=torch.float32)

#                 reinforcement_update_batch(user_tensor, item_tensor, label_tensor)

#                 db.execute(text("DELETE FROM rl_buffer"))
#                 db.commit()
#             finally:
#                 db.close()


#         return f"✅ Feedback recorded for course {course_id}"

#     except Exception as e:
#         return f"❌ Error saving feedback: {e}"


# @dash.callback(
#     [Output("recipe-modal", "is_open"),
#      Output("modal-title", "children"),
#      Output("modal-content", "children")],
#     [Input({"type": "view-more-btn", "index": ALL}, "n_clicks"),
#      Input("close-modal", "n_clicks")],
#     [State("recipe-modal", "is_open")],
#     prevent_initial_call=True
# )


# def toggle_modal(view_clicks, close_clicks, is_open):
#     if ctx.triggered_id == "close-modal":
#         return False, "", None
    
#     if not any(view_clicks):
#         raise dash.exceptions.PreventUpdate
    
#     triggered_id = ctx.triggered_id
#     if triggered_id:
#         course_id = triggered_id["index"]
#         course = course_df[course_df["course_id"] == course_id].iloc[0]
        
#         modal_content = dbc.Container([
#             dbc.Row([
#                 dbc.Col([
#                     html.Img(
#                         src=course["image_url"],
#                         style={
#                             "width": "100%",
#                             "borderRadius": "10px",
#                             "marginBottom": "20px"
#                         }
#                     ),
#                     html.Div([
#                         html.H5("Details", className="mb-3", style={"textAlign": "left"}),  # Add textAlign style
#                         dbc.Row([
#                             # Left column
#                             dbc.Col([
#                                 html.P(f"Category: {course['category'].title()}", className="mb-2"),
#                                 html.P(f"Calories: {course['calories']:.2f}", className="mb-2"),
#                             ], width=6),
#                             # Right column
#                             dbc.Col([
#                                 html.P(f"Sugar: {course['sugar']:.2f}g", className="mb-2"),
#                                 html.P(f"Fiber: {course['fiber']:.2f}g", className="mb-2"),
#                             ], width=6),
#                         ]),
#                     ], className="mb-4"),
#                 ], width=12, lg=6),
#                 dbc.Col([
#                     html.Div([
#                         html.H5("Ingredients", className="mb-3"),
#                         html.P(course["ingredients"], className="mb-4", 
#                               style={"whiteSpace": "pre-line"}),
#                         html.H5("Recipe", className="mb-3"),
#                         html.P(course["cooking_directions"], 
#                               style={"whiteSpace": "pre-line"})
#                     ])
#                 ], width=12, lg=6)
#             ])
#         ], fluid=True, className="px-0")
        
#         return True, course["course_name"], modal_content
    
#     return False, "", None

# @dash.callback(
#     Output({"type": "collapse", "index": MATCH}, "is_open"),
#     Input({"type": "view-more-btn", "index": MATCH}, "n_clicks"),
#     State({"type": "collapse", "index": MATCH}, "is_open"),
#     prevent_initial_call=True
# )
# def toggle_collapse(n_clicks, is_open):
#     if n_clicks:
#         return not is_open
#     return is_open







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


def load_course_df():
    print("🌍 RAILWAY_ENVIRONMENT:", os.environ.get("RAILWAY_ENVIRONMENT"))
    file_id = os.environ.get("COURSE_CSV_DRIVE_ID")
    
    # Only download if running on Railway AND the file doesn't exist yet
    if os.environ.get("RAILWAY_ENVIRONMENT") and file_id and not os.path.exists("course_cleaned.csv"):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"⬇️ Downloading from: {url}")
        import gdown
        gdown.download(url, "course_cleaned.csv", quiet=False)
    
    # ✅ After ensuring it's downloaded, load only selected columns
    return pd.read_csv(
        "course_cleaned.csv",
        usecols=["course_id", "course_name", "image_url", "category", "calories", "sugar", "fiber"]
    )


# ✅ Call the loader
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
# course_df = pd.read_csv("course_cleaned.csv")
course_df["category"] = course_df["category"].str.lower()
scaler = joblib.load("trained_scaler.pkl")

# if "user_index" not in df.columns:
#         # Load sugar level history from DB
#     db = SessionLocal()
#     try:
#         query = text("SELECT sugar_value, timestamp FROM sugar_levels WHERE user_id = :user_id ORDER BY timestamp ASC")
#         result = db.execute(query, {"user_id": user_id}).fetchall()
#         sugar_df = pd.DataFrame(result, columns=["sugar_value", "timestamp"])
#     finally:
#         db.close()

# unique_users = df["user_id"].unique()
# user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
# df["user_index"] = df["user_id"].map(user_mapping)

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

# def classify_sugar_level(sugar_value):
#     sugar_value = float(sugar_value)
#     if sugar_value > 180:
#         return "high"
#     elif 100 <= sugar_value <= 180:
#         return "normal"
#     else:
#         return "low"

# Dash layout (no login/signup here anymore!)

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
        dcc.Store(id="session-bmi", storage_type="session"),  # BMI

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
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
            },
            className="modal-dialog-centered modal-dialog-scrollable"
        ),
    ], fluid=True, className="dashapp1-content")
], className="dashapp1-container")

# Display recommendations using session values
@dash.callback(
    Output("recommendation-section", "children"),
    Input("session-user-id", "data"),
    Input("session-sugar", "data"),
    Input("session-bmi", "data")  #BMI added
)
def display_recommendations(user_id, sugar_value, bmi_value):#bmi added
    
    if user_id is None or sugar_value is None:
        return dbc.Alert("You must be signed in to see recommendations.", color="danger")

    # user_mapping = df[["user_id", "user_index"]].drop_duplicates()
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

    # def get_model_recommendations(user_index, top_n=50):
    #     user_tensor = torch.tensor([user_index], dtype=torch.long)
    #     item_tensor = torch.arange(num_items, dtype=torch.long)
    #     with torch.no_grad():
    #         scores = model(user_tensor.repeat(num_items), item_tensor).squeeze()
    #     sorted_indices = torch.argsort(scores, descending=True).tolist()
    #     return course_df.iloc[sorted_indices[:top_n]]

    # def get_filtered_recommendations(user_index, sugar_value, bmi_value, top_n=10):
    #     sugar_category = classify_sugar_level(sugar_value)
    #     thresholds = sugar_thresholds[sugar_category]
    #     recs = course_df.copy() if is_new_user else get_model_recommendations(user_index)
    #     if is_new_user and bmi_value is not None: #added new
    #         bmi_value = float(bmi_value)
    #         if bmi_value >= 30: #added new
    #             thresholds["max_calories"] -= 100
    #             thresholds["max_sugar"] -= 5
    #             thresholds["min_fiber"] += 0.5
    #     filtered = recs[
    #         (recs["sugar"] <= thresholds["max_sugar"]) &
    #         (recs["calories"] <= thresholds["max_calories"]) &
    #         (recs["fiber"] >= thresholds["min_fiber"])
    #     ]
    #     return filtered.head(top_n)

    # filtered = get_filtered_recommendations(user_index, sugar_value, bmi_value)

    # if (filtered.empty):
    #     return html.P("No meals match your health preferences.")
    # # Load sugar level history from DB
    # db = SessionLocal()
    # try:
    #     query = text("SELECT sugar_value, timestamp FROM sugar_levels WHERE user_id = :user_id ORDER BY timestamp ASC")
    #     result = db.execute(query, {"user_id": user_id}).fetchall()
    #     sugar_df = pd.DataFrame(result, columns=["sugar_value", "timestamp"])
    # finally:
    #     db.close()

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

    # if "user_index" not in df.columns:
    #         # Load sugar level history from DB
    #     db = SessionLocal()
    #     try:
    #         query = text("SELECT sugar_value, timestamp FROM sugar_levels WHERE user_id = :user_id ORDER BY timestamp ASC")
    #         result = db.execute(query, {"user_id": user_id}).fetchall()
    #         sugar_df = pd.DataFrame(result, columns=["sugar_value", "timestamp"])
    #     finally:
    #         db.close()
    # ✅ Always load user's sugar data here
    db = SessionLocal()
    try:
        query = text("SELECT sugar_value, timestamp FROM sugar_levels WHERE user_id = :user_id ORDER BY timestamp ASC")
        result = db.execute(query, {"user_id": user_id}).fetchall()
        sugar_df = pd.DataFrame(result, columns=["sugar_value", "timestamp"])
    finally:
        db.close()


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
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            title_font_size=18,
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)'  # Light grey grid with 50% opacity
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(211, 211, 211, 0.5)'  # Light grey grid with 50% opacity
            )
        )
        sugar_chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    cards = []
    for _, row in filtered.iterrows():
        course_id = row["course_id"]
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    # Image first
                    dbc.CardImg(
                        src=row["image_url"], 
                        top=True,
                        style={
                            "width": "100%",
                            "height": "200px",
                            "object-fit": "cover",
                            "borderRadius": "10px"  # Added border radius
                        }
                    ),
                    # Title second
                    html.H6(
                        row["course_name"], 
                        className="card-title mb-3",  # Removed text-center class
                        style={
                            "padding-top":"10px",
                            "textAlign": "left"  # Explicitly set left alignment
                        }
                    ),
                    # Descriptions third
                    html.Div([
                        html.Span(f"{row['category'].title()}", className="mb-2"),
                        # html.P(f" Calories: {row['calories']:.2f}", className="mb-2"),
                        # html.P(f" Sugar: {row['sugar']:.2f}g", className="mb-2"),
                        # html.P(f" Fiber: {row['fiber']:.2f}g", className="mb-2"),
                        # html.P(f" Ingredients: {row['ingredients'][:100]}...", className="mb-2"),  # Shortened preview
                        # html.P(f" Recipe: {row['cooking_directions'][:100]}...", className="mb-2"),  # Shortened preview
                    ], className="card-text"),

                    # Replace the View More button and ButtonGroup section with:
                    html.Div([
                        # Left side - View More button
                        dbc.Button(
                            "View More",
                            id={"type": "view-more-btn", "index": int(course_id)},
                            color="link",
                            className="text-primary p-0",
                            style={
                                "textDecoration": "none",
                                "fontSize": "14px",
                                "fontWeight": "500"
                            }
                        ),
                        # Right side - Like/Dislike buttons
                        dbc.ButtonGroup([
                            dbc.Button(
                                html.I(className="far fa-heart"), 
                                id={"type": "like-btn", "index": int(course_id)}, 
                                color="link",
                                className="text-dark",
                                size="sm",
                                style={
                                    "border": "none",
                                    "background": "none",
                                    "padding": "0px",
                                    "marginRight": "10px",
                                    "fontSize": "20px"
                                }
                            ),
                            dbc.Button(
                                html.I(className="far fa-thumbs-down"),
                                id={"type": "dislike-btn", "index": int(course_id)}, 
                                color="link",
                                className="text-dark",
                                size="sm",
                                style={
                                    "border": "none",
                                    "background": "none",
                                    "padding": "0px",
                                    "fontSize": "20px"
                                }
                            )
                        ])
                    ], className="d-flex justify-content-between align-items-center mb-2")
                ], className="p-4", 
                style={
                    "backgroundColor": "#faf7f2",  # Light beige color
                    "border": "1px solid #e9ecef",
                    "borderRadius": "15px",
                    "boxShadow": "none"
                })
            ], className="mb-4 h-100 card-hover",
            style={
                "backgroundColor": "#faf7f2",
                "border": "1px solid #e9ecef",
                "borderRadius": "15px",
                "boxShadow": "none"
            })
        )
    return html.Div([
        # Carousel of recommendations first
        html.Div(
            id="carousel-wrapper",
            children=[
                html.Div(
                    card,
                    style={
                        "minWidth": "300px",
                        "maxWidth": "300px",
                        "marginRight": "15px",
                        "flex": "0 0 auto",
                        "scrollSnapAlign": "start"
                    }
                ) for card in cards
            ],
            style={
                "display": "flex",
                "overflowX": "auto",
                "scrollBehavior": "smooth",
                "padding": "10px",
                "gap": "10px",
                "scrollSnapType": "x mandatory",
                "msOverflowStyle": "none",
                "scrollbarWidth": "none",
                "::-webkit-scrollbar": {
                    "display": "none"
                }
            },
            className="carousel-container"
        ),

        # Scroll buttons
        html.Div([
            dbc.Button(
                html.I(className="fas fa-chevron-left"),
                id="scroll-left",
                color="light",
                className="me-2 rounded-circle",
                style={
                    "width": "40px",
                    "height": "40px",
                    "padding": "0",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "backgroundColor": "#ffffff",
                    "border": "1px solid #dee2e6",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                }
            ),
            dbc.Button(
                html.I(className="fas fa-chevron-right"),
                id="scroll-right",
                color="light",
                className="rounded-circle",
                style={
                    "width": "40px",
                    "height": "40px",
                    "padding": "0",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "backgroundColor": "#ffffff",
                    "border": "1px solid #dee2e6",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                }
            )
        ], className="d-flex mt-3", style={"position": "fixed", "right": "80px"}),

        # Sugar chart moved to bottom with padding
        html.Div([
            html.Hr(className="mt-5"),  # Add a divider
            sugar_chart
        ], className="mt-5 mb-4"),  # Add top and bottom margin

        dcc.Store(id="carousel-scroll")
    ])
# def reinforcement_update_batch(user_tensor, item_tensor, label_tensor):
#     print(f"🧠 Starting batch training on {len(user_tensor)} samples...")
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.BCELoss()

#     optimizer.zero_grad()
#     predictions = model(user_tensor, item_tensor)
#     loss = criterion(predictions, label_tensor)
#     loss.backward()
#     optimizer.step()

#     model.eval()
#     print("✅ Reinforcement batch update done")

# Handle likes and dislikes
@dash.callback(
    Output("feedback-msg", "children", allow_duplicate=True),
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

        # 🧠 Compute NeuralFM-compatible indices (safe for new users)
        # user_mapping = df[["user_id", "user_index"]].drop_duplicates()
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT DISTINCT user_id, user_index FROM filtered_df WHERE user_index IS NOT NULL"))
            user_mapping = pd.DataFrame(result.fetchall(), columns=["user_id", "user_index"])
        finally:
            db.close()


        if user_id in user_mapping["user_id"].values:
            user_index = user_mapping[user_mapping["user_id"] == user_id]["user_index"].values[0]
        else:
            # user_index = df["user_index"].max() + 1 if not df.empty else 0
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

        # new_row = pd.DataFrame([{
        #     "user_id": user_id,
        #     "name": user_name,
        #     "course_id": course_id,
        #     "user_index": user_index,  # ✅ NOW this is defined correctly
        #     "rating": rating,
        #     "course_name": course_row["course_name"],
        #     "category": category_code,
        #     "image_url": course_row["image_url"],
        #     "ingredients": course_row["ingredients"],
        #     "cooking_directions": course_row["cooking_directions"],
        #     "calories": calories_norm,
        #     "sugar": sugar_norm,
        #     "fiber": fiber_norm
        # }])
        # new_row["user_index"] = user_index

        # 🧠 Compute NeuralFM-compatible indices
        # user_mapping = df[["user_id", "user_index"]].drop_duplicates()
        # user_index = user_mapping[user_mapping["user_id"] == user_id]["user_index"].values[0]
        # 🧠 Compute NeuralFM-compatible indices (safe for new users)
    
        new_row_csv = pd.DataFrame([{
            "user_id": user_id,
            "name": user_name,
            "email": "",  # or actual
            "password": "",  # or actual
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
        # # ✅ Insert into RL buffer
        # # ✅ Save to CSV and DB
        # new_row.to_csv(file_path, mode='a', header=False, index=False)
        # conn = sqlite3.connect("meals.db")
        # cursor = conn.cursor()
        # new_row.to_sql("filtered_df", conn, if_exists="append", index=False)
        # ✅ Save to CSV (includes email, password)
        new_row_csv.to_csv(file_path, mode='a', header=False, index=False)
        # ✅ Save to SQLite (excludes email, password)
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
            # db.execute(insert_rl, {
            #     "user_index": user_index,
            #     "course_index": course_index,
            #     "rating": rating
            # })
            db.execute(insert_rl, {
                "user_index": int(user_index),
                "course_index": int(course_index),
                "rating": int(rating)
            })
            print("✅ handle_rating triggered - user_id:", user_id)


            db.commit()
            count_result = db.execute(text("SELECT COUNT(*) FROM rl_buffer"))
            count = count_result.scalar()
        finally:
            db.close()

        # ✅ Only enter this block if needed
        if count >= 50:
            db = SessionLocal()  # New session
            try:
                df_buffer = pd.read_sql("SELECT * FROM rl_buffer", db.connection())

                user_tensor = torch.tensor(df_buffer["user_index"].astype(int).values, dtype=torch.long)
                item_tensor = torch.tensor(df_buffer["course_index"].astype(int).values, dtype=torch.long)
                label_tensor = torch.tensor(df_buffer["rating"].astype(float).values, dtype=torch.float32)

                reinforcement_update_batch(user_tensor, item_tensor, label_tensor)

                db.execute(text("DELETE FROM rl_buffer"))
                db.commit()
            finally:
                db.close()


        return f"✅ Feedback recorded for course {course_id}"

    except Exception as e:
        return f"❌ Error saving feedback: {e}"


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
                        style={
                            "width": "100%",
                            "borderRadius": "10px",
                            "marginBottom": "20px"
                        }
                    ),
                    html.Div([
                        html.H5("Details", className="mb-3", style={"textAlign": "left"}),  # Add textAlign style
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
                        html.P(course["ingredients"], className="mb-4", 
                              style={"whiteSpace": "pre-line"}),
                        html.H5("Recipe", className="mb-3"),
                        html.P(course["cooking_directions"], 
                              style={"whiteSpace": "pre-line"})
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

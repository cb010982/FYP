import torch
import pandas as pd
from dash import html

# Define thresholds (can be moved to a config module)
sugar_thresholds = {
    "high": {"max_sugar": 15, "max_calories": 350, "min_fiber": 2},
    "normal": {"max_sugar": 30, "max_calories": 500, "min_fiber": 1.5},
    "low": {"max_sugar": 40, "max_calories": 700, "min_fiber": 1}
}

def classify_sugar_level(sugar_value):
    sugar_value = float(sugar_value)
    if sugar_value > 180:
        return "high"
    elif 100 <= sugar_value <= 180:
        return "normal"
    else:
        return "low"

def get_model_recommendations(user_index, model, course_df, num_items, top_n=50):
    user_tensor = torch.tensor([user_index], dtype=torch.long)

    # Ensure item_tensor stays within embedding bounds
    max_index = model.item_embedding.num_embeddings
    if num_items > max_index:
        num_items = max_index
    item_tensor = torch.arange(num_items, dtype=torch.long)

    with torch.no_grad():
        scores = model(user_tensor.repeat(num_items), item_tensor).squeeze()
    sorted_indices = torch.argsort(scores, descending=True).tolist()
    return course_df.iloc[sorted_indices[:top_n]]


def get_filtered_recommendations(user_index, sugar_value, bmi_value, is_new_user, course_df, model, num_items, top_n=10):
    sugar_category = classify_sugar_level(sugar_value)
    thresholds = sugar_thresholds[sugar_category]
    recs = course_df.copy() if is_new_user else get_model_recommendations(user_index, model, course_df, num_items, top_n=top_n)

    if is_new_user and bmi_value is not None:
        bmi_value = float(bmi_value)
        if bmi_value >= 30:
            thresholds["max_calories"] -= 100
            thresholds["max_sugar"] -= 5
            thresholds["min_fiber"] += 0.5

    filtered = recs[
        (recs["sugar"] <= thresholds["max_sugar"]) &
        (recs["calories"] <= thresholds["max_calories"]) &
        (recs["fiber"] >= thresholds["min_fiber"])
    ]

    return filtered.head(top_n)

def reinforcement_update_batch(model, user_tensor, item_tensor, label_tensor):
    print(f"🧠 Starting batch training on {len(user_tensor)} samples...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    optimizer.zero_grad()
    predictions = model(user_tensor, item_tensor)
    loss = criterion(predictions, label_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    print("✅ Reinforcement batch update done")

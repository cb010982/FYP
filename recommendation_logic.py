from sqlalchemy import text
import torch
import pandas as pd
from dash import html
from sqlalchemy import text
from db import SessionLocal


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

def get_filtered_recommendations(
    user_index, sugar_value, bmi_value, is_new_user, course_df, model, num_items, top_n=10
):
 

    sugar_category = classify_sugar_level(sugar_value)
    thresholds = sugar_thresholds[sugar_category].copy()

    if bmi_value is not None:
        bmi_value = float(bmi_value)
        if bmi_value >= 30:
            thresholds["max_calories"] -= 100
            thresholds["max_sugar"] -= 5
            thresholds["min_fiber"] += 0.5

    #  Prevent crash for new users with user_index = None
    if is_new_user and user_index is not None:
        db = SessionLocal()
        try:
            result = db.execute(text("""
                SELECT COUNT(*) FROM filtered_df
                WHERE user_index = :user_index
            """), {"user_index": int(user_index)})
            count = result.scalar()
            if count > 0:
                is_new_user = False
        finally:
            db.close()

    # Step 2: Get base recommendations
    recs = course_df.copy() if is_new_user else get_model_recommendations(
        user_index, model, course_df, num_items, top_n=100
    )

    # Step 3: Apply filters
    filtered = recs[
        (recs["sugar"] <= thresholds["max_sugar"]) &
        (recs["calories"] <= thresholds["max_calories"]) &
        (recs["fiber"] >= thresholds["min_fiber"])
    ]

    # Step 4: Filter out liked/disliked if returning user
    if not is_new_user and user_index is not None:
        db = SessionLocal()
        try:
            result = db.execute(text("""
                SELECT course_id, rating FROM filtered_df
                WHERE user_index = :user_index
            """), {"user_index": int(user_index)})

            interactions = result.fetchall()
            disliked_ids = [row[0] for row in interactions if row[1] == 0]
            liked_ids = [row[0] for row in interactions if row[1] == 1]

            #  Convert course_id column in DataFrame to same type as DB IDs
            filtered["course_id"] = filtered["course_id"].astype(str)  # or int, match your DB
            disliked_ids = [str(x) for x in disliked_ids]  # match the DataFrame type
            liked_ids = [str(x) for x in liked_ids]

            filtered = filtered[~filtered["course_id"].isin(disliked_ids + liked_ids)]
        finally:
            db.close()

    return filtered.head(top_n)


def reinforcement_update_batch(model, user_tensor, item_tensor, label_tensor):
    print(f" Starting batch training on {len(user_tensor)} samples...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    optimizer.zero_grad()
    predictions = model(user_tensor, item_tensor)
    loss = criterion(predictions, label_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    print(" Reinforcement batch update done")


    # torch.save(model.user_embedding.state_dict(), "nfm_user_embedding.pth")
    # torch.save(model.item_embedding.state_dict(), "nfm_item_embedding.pth")
    # torch.save(model.fm_layer.state_dict(), "nfm_fm_layer.pth")
    # torch.save(model.dnn.state_dict(), "nfm_dnn.pth")

    # model.user_embedding.load_state_dict(torch.load("nfm_user_embedding.pth"))
    # model.item_embedding.load_state_dict(torch.load("nfm_item_embedding.pth"))
    # model.fm_layer.load_state_dict(torch.load("nfm_fm_layer.pth"))
    # model.dnn.load_state_dict(torch.load("nfm_dnn.pth"))
    # model.eval()



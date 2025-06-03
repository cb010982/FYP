import dash
from dash import dcc, html, Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import numpy as np
import joblib
from dash.exceptions import PreventUpdate

# Load model
rf_model = joblib.load("diabetes_random_forest_model.pkl")

# Define questions
questions = [
    {"id": "pregnancies", "label": "Number of pregnancies you had?", "min": 0, "max": 20, "default": 1},
    {"id": "glucose", "label": "What is your current glucose level?", "min": 0, "max": 200, "default": 120},
    {"id": "blood_pressure", "label": "What is your current blood pressure?", "min": 0, "max": 150, "default": 70},
    {"id": "skin_thickness", "label": "Measure of your skin thickness", "min": 0, "max": 100, "default": 20},
    {"id": "insulin", "label": "What is your current insulin level?", "min": 0, "max": 900, "default": 79},
    {"id": "bmi", "label": "What is your BMI?", "min": 0, "max": 70, "default": 25.0},
    {"id": "dpf", "label": "What is your Diabetes Pedigree Function level at?", "min": 0, "max": 2.5, "default": 0.5},
    {"id": "age", "label": "Your Age?", "min": 0, "max": 120, "default": 30}
]

def create_question_card(question_idx):
    question = questions[question_idx]
    return html.Div([
        html.H5(question["label"], className="mb-4"),
        # Custom number input with + and - buttons
        html.Div([
            dbc.Button("-", 
                id={"type": "decrease-btn", "index": question_idx},
                color="light",
                className="number-btn"
            ),
            dbc.Input(
                id={"type": "question-input", "index": question_idx},
                type="number",
                min=question["min"],
                max=question["max"],
                value=question["default"],
                className="number-input text-center",
                readonly=True
            ),
            dbc.Button("+", 
                id={"type": "increase-btn", "index": question_idx},
                color="light",
                className="number-btn"
            ),
        ], className="number-input-group")
    ], className="question-content")

# Update the layout section
layout = html.Div([
    # Header
    html.H4(" Let's check your diabetes risk", className="text-center mt-4"),
    # html.P("Let's check your diabetes risk step by step", className="text-center mb-4"),
    
    # Progress dots only
    html.Div([
        html.Div([
            html.Div([
                html.Div(id={"type": "progress-dot", "index": i}, 
                        className="progress-dot")
                for i in range(len(questions))
            ], className="progress-dots")
        ], className="progress-container")
    ], className="progress-bar-container"),
    
    # Card container
    html.Div([
        # Question card
        html.Div(id="question-card", className="question-card"),
        
        # Navigation buttons
        html.Div([
            dbc.Button(
                html.I(className="fas fa-chevron-left"),
                id="prev-btn",
                color="light",
                className="nav-button prev",
            ),
            dbc.Button(
                html.I(className="fas fa-chevron-right"),
                id="next-btn",
                color="success",
                className="nav-button next"
            )
        ], className="nav-buttons")
    ], className="card-container"),
    
    # Store components
    dcc.Store(id="current-step", data=0),
    dcc.Store(id="answers-store", data=[q["default"] for q in questions]),
    dcc.Store(id="temp-glucose-value", storage_type="session"),  # Added temp storage for glucose
    dcc.Store(id="temp-bmi-value", storage_type="session"),  # NEW BMI store

    
    # Prediction output
    html.Div(id="prediction-output", className="mt-4")
], className="prediction-container")

@dash.callback(
    Output("answers-store", "data"),
    [Input({"type": "question-input", "index": ALL}, "value")],
    [State("answers-store", "data"),
     State("current-step", "data")]
)
def store_answer(values, current_answers, current_step):
    if not dash.callback_context.triggered:
        raise PreventUpdate
    
    if values and len(values) > 0:
        current_answers[current_step] = values[0]
    return current_answers

# Update the callback to handle progress dots instead of progress bar
@dash.callback(
    [Output("question-card", "children"),
     Output({"type": "progress-dot", "index": ALL}, "className"),
     Output("prev-btn", "disabled"),
     Output("next-btn", "children"),
     Output("current-step", "data")],
    [Input("prev-btn", "n_clicks"),
     Input("next-btn", "n_clicks")],
    [State("current-step", "data"),
     State("answers-store", "data")]
)
def update_question(prev_clicks, next_clicks, current_step, answers):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        dot_classes = ["progress-dot active" if i == 0 else "progress-dot" 
                      for i in range(len(questions))]
        return create_question_card(0), dot_classes, True, "Next", 0
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "prev-btn" and current_step > 0:
        current_step -= 1
    elif button_id == "next-btn" and current_step < len(questions) - 1:
        current_step += 1
    
    # Update progress dots
    dot_classes = ["progress-dot active" if i <= current_step else "progress-dot" 
                  for i in range(len(questions))]
    
    prev_disabled = current_step == 0
    next_text = "Predict" if current_step == len(questions) - 1 else html.I(className="fas fa-chevron-right")
    
    return create_question_card(current_step), dot_classes, prev_disabled, next_text, current_step

# Modify the predict_diabetes callback to store the glucose value
@dash.callback(
    [Output("prediction-output", "children"),
     Output("temp-glucose-value", "data"),
     Output("temp-bmi-value", "data")], #added bmi to return
    Input("next-btn", "n_clicks"),
    [State("answers-store", "data"),
     State("current-step", "data")]
)
def predict_diabetes(n_clicks, answers, current_step):
    if not n_clicks or current_step != len(questions) - 1:
        raise PreventUpdate
    
    try:
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age = answers
        
        if glucose >= 97:
            return html.Div([
                dbc.Alert(
                    f"High Risk Detected. Wanna get proper diet recommendations to help get back on track?", 
                    color="danger"
                ),
                html.Div([
                    dbc.Button(
                        "Get Diet Recommendations!", 
                        href="/login",
                        color="success", 
                        className="mt-3"
                    )
                ], className="text-center")
            ], className="prediction-result"), glucose, bmi #added bmi to return
        
        user_input = np.array([answers])
        prediction = rf_model.predict(user_input)[0]
        prob = rf_model.predict_proba(user_input)[0, 1]
        
        if prediction == 1:
            return html.Div([
                dbc.Alert(
                    f"The model predicts diabetes with a probability of {prob:.2f}",
                    color="danger"
                ),
                html.Div([
                    dbc.Button(
                        "Proceed to Get Diet Recommendations",  # Updated button text
                        href="/login",  # Changed from /signin to /dashapp1
                        color="success",
                        className="mt-3"
                    )
                ], className="text-center")
            ], className="prediction-result"), glucose, bmi #added bmi to return
        else:
            return html.Div([
                dbc.Alert(
                    f"The model predicts no diabetes with a probability of {1 - prob:.2f}",
                    color="success"
                )
            ], className="prediction-result"), glucose, bmi #added bmi to return
            
    except Exception as e:
        return html.Div([
            dbc.Alert(
                f"Error in prediction: {str(e)}",
                color="danger"
            )
        ], className="prediction-result"), None

# Add new callback for + and - buttons
@dash.callback(
    Output({"type": "question-input", "index": MATCH}, "value"),
    [Input({"type": "increase-btn", "index": MATCH}, "n_clicks"),
     Input({"type": "decrease-btn", "index": MATCH}, "n_clicks")],
    [State({"type": "question-input", "index": MATCH}, "value"),
     State({"type": "question-input", "index": MATCH}, "min"),
     State({"type": "question-input", "index": MATCH}, "max")]
)
def update_number(increase_clicks, decrease_clicks, current_value, min_value, max_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if "increase" in button_id and current_value < max_value:
        return current_value + 1
    elif "decrease" in button_id and current_value > min_value:
        return current_value - 1
    
    return current_value

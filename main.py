import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dashapp import layout as dashapp_layout
from dashapp1 import layout as dashapp1_layout  
from dash.dependencies import Input, Output
from login import layout as login_layout
from signup import layout as signup_layout
from login import init_callbacks as init_login_callbacks
from signup import init_callbacks as init_signup_callbacks


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
    ],
    suppress_callback_exceptions=True
)
server = app.server

app.title = "Landing Page"

init_login_callbacks(app)
init_signup_callbacks(app)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")  
])

# Define navigation bar component
nav_bar = html.Nav([
    html.Div([
        html.Span("EatSafe", className="brand"),

        # Desktop Nav links
        html.Ul([
            html.Li(html.A("Home", href="/", className="active")),
            html.Li(html.A("Diabetes Prediction", href="/dashapp")),
            html.Li(html.A("Get Started", href="/dashapp")),
            html.Li(html.A("Login", href="/login")),
            html.Li(html.A("Begin", href="/dashapp")),
        ], className="nav-links full-nav"),

        # Mobile nav links
        html.Div([
            html.A("Login", href="/login", className="mobile-login-link", style={"display": "block", "width": "100%", "textAlign": "center"}),
            html.A("Get Started", href="/dashapp", className="mobile-get-started-btn", style={"display": "block", "width": "100%", "textAlign": "center"})
        ], className="mobile-nav")
    ], className="nav-container")
], className="glass-navbar")

# Home page layout with navigation
home_layout = html.Div([
    nav_bar,  # Include navigation only on home page
    html.Div([
        html.Div([
            html.H1("Enjoy a Happy, Healthy Diet!", className="display-4 fw-bold mb-3"),
            html.P("Let's control diabetes through dietary interventions!", className="lead"),
            html.Div([
                dbc.Button("Get Started", color="danger", href="/dashapp")
            ])
        ], className="hero-text")
    ], className="hero-section")
])

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/dashapp":
        return dashapp_layout
    elif pathname == "/dashapp1":
        return dashapp1_layout
    elif pathname == "/login":
        return login_layout
    elif pathname == "/signup":
        return signup_layout
    elif pathname == "/" or pathname == "":
        return home_layout
    else:
        return html.H1("404 - Page not found", className="text-center")
    
app.clientside_callback(
    """
    function(nLeft, nRight) {
        var container = document.querySelector('#carousel-wrapper');
        if (!container) return window.dash_clientside.no_update;

        const scrollAmount = 320;

        if (nLeft > 0) {
            container.scrollBy({ left: -scrollAmount, behavior: 'smooth' });
        } else if (nRight > 0) {
            container.scrollBy({ left: scrollAmount, behavior: 'smooth' });
        }
        return null;
    }
    """,
    Output("carousel-scroll", "data"),
    Input("scroll-left", "n_clicks"),
    Input("scroll-right", "n_clicks"),
    prevent_initial_call=True
)

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)

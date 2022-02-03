import dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, TriggerTransform, \
    MultiplexerTransform

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = DashProxy(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True,
                transforms=[TriggerTransform(), MultiplexerTransform()])

# app = dash.Dash(__name__, suppress_callback_exceptions=True)
# server = app.server

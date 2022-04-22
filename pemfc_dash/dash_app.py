import dash
import dash_bootstrap_components as dbc
from dash.long_callback import CeleryLongCallbackManager
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from celery import Celery

celery_app = Celery(
    __name__, broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1")
long_callback_manager = CeleryLongCallbackManager(celery_app)

# app = dash.Dash(__name__, long_callback_manager=long_callback_manager)

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
bs_4_css = ('https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css'
              '/bootstrap.min.css')
bs_5_css = ('https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css')


external_stylesheets = [bs_5_css]
app = DashProxy(__name__, external_stylesheets=external_stylesheets,
                long_callback_manager=long_callback_manager,
                suppress_callback_exceptions=True,
                transforms=[MultiplexerTransform()])

# app = dash.Dash(__name__, suppress_callback_exceptions=True)
# server = app.server

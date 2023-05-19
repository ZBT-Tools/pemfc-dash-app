from dash_extensions.enrich import html, dcc
import dash_bootstrap_components as dbc
from . import modal_functions as mf
from pemfc_dash.dash_app import app

server = app.server

app._favicon = 'logo-zbt.ico'
app.title = 'PEMFC Model'

# Import layout components
from pemfc_dash.layout.header import header
from pemfc_dash.layout.menu_column import menu_column
from pemfc_dash.layout.result_column import result_column
from pemfc_dash.layout.bottom import bottom


# App layout
# -----------------------------------------------------------------------------
app.layout = dbc.Container([
    # Register any non-visible divs for caching data/files and signaling
    dcc.Store(id="pemfc_settings_file"),
    dcc.Store(id="input_data"),
    dcc.Store(id="df_input_data"),
    dbc.Spinner(dcc.Store(id='result_data_store'), fullscreen=True,
                spinner_class_name='loading_spinner',
                fullscreen_class_name='loading_spinner_bg'),
    dcc.Store(id='df_result_data_store'),
    dcc.Store(id='df_input_store'),
    dcc.Store(id='variation_parameter'),
    dcc.Store(id='cells_data'),
    dcc.Store(id='modal_store'),
    # Dummy div for initialization
    # (Read available input parameters, create study table)
    html.Div(id="initial_dummy"),

    # empty Div to trigger javascript file for graph resizing
    html.Div(id="output-clientside"),
    # Modal for any warning
    mf.create_modal(),

    # Load html elements for header row
    header,
    # Load html elements for left menu and right results columns
    html.Div([  # MIDDLE
        menu_column,
        result_column
        ],
        className="row",
        style={'justify-content': 'space-evenly'}),
    # Load html elements for bottom row
    bottom],

    id="mainContainer",
    # className='twelve columns',
    fluid=True,
    style={'padding': '0px'})

# Import callback functions
# -----------------------------------------------------------------------------
from . import callbacks

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)

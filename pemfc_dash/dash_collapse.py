import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

from dash_app import app

from dash.dependencies import Input, Output, State, ALL  # ClientsideFunction


collapses = \
    html.Div([
        dbc.Collapse(html.Div(
            [dbc.Spinner(
                dcc.Upload(id='upload-file',
                           children=html.Div(
                                    ['Drag and Drop or ',
                                     html.A('Select Files',
                                            style=
                                            {'font-weight': 'bold',
                                             'text-decoration': 'underline'})]),
                           className='dragndrop'), fullscreen=True,
                fullscreen_class_name='loading_spinner_bg',
                spinner_class_name='loading_spinner')]),
            # accept='.json',
            id='collapse_load', is_open=False),
        dbc.Collapse(html.Div(
            [html.Div('Save setting as  : ', className='label-param-s'),
             dbc.Input(id='save_as_input', placeholder="settings.json",
                       className='label-param-l'),
             html.Button('Go', id='save-as-button',
                         className='ufontm centered')], className='r_flex g-0'),
            id='collapse_save', is_open=False)])


@app.callback(
    Output("collapse_load", "is_open"),
    Output("collapse_save", "is_open"),
    Input("load-button", "n_clicks"),
    Input("save-button", "n_clicks"),
    State("collapse_load", "is_open"),
    State("collapse_save", "is_open"),
)
def toggle_collapse(n, n1, is_open, is_open2):
    ctx = dash.callback_context.triggered[0]['prop_id']
    if n or n1:
        if ctx == 'save-button.n_clicks':
            if is_open is True:
                return not is_open, not is_open2
            return is_open, not is_open2
        else:
            if is_open2 is True:
                return not is_open, not is_open2
            return not is_open, is_open2
    return is_open, is_open2

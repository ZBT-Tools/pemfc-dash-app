import base64
import gc
import io
import os
from itertools import product
import ast
from dash import dash_table
import numpy as np
import pandas as pd
import pickle
import re
import copy
import sys
import json

import dash
from dash_extensions.enrich import Output, Input, State, ALL, html, dcc, \
    ServersideOutput, ctx
from dash import dash_table as dt
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from pemfc_dash.dash_functions import create_settings
from . import dash_functions as df, dash_layout as dl, \
    dash_modal as dm
from pemfc_dash.dash_app import app

import pemfc
from pemfc.src import interpolation as ip
from pemfc import main_app
from pemfc_gui import data_transfer
import pemfc_gui.input as gui_input

from pandarallel import pandarallel
from pemfc_dash.study_functions import uicalc_prepare_initcalc, uicalc_prepare_refinement
from tqdm import tqdm

tqdm.pandas()
# pandarallel.initialize()
from multiprocesspandas import applyparallel

server = app.server

app._favicon = 'logo-zbt.ico'
app.title = 'PEMFC Model'

# Component Initialization & App layout
# ----------------------------------------

# Process bar components
pbar = dbc.Progress(id='pbar')
timer_progress = dcc.Interval(id='timer_progress',
                              interval=15000)

app.layout = dbc.Container([
    dcc.Store(id="pemfc_settings_file"),
    dcc.Store(id="input_data"),
    dcc.Store(id="df_input_data"),
    dbc.Spinner(dcc.Store(id='result_data_store'), fullscreen=True,
                spinner_class_name='loading_spinner',
                fullscreen_class_name='loading_spinner_bg'),
    dcc.Store(id='df_result_data_store'),
    dcc.Store(id='df_input_store'),
    dcc.Store(id='variation_parameter'),
    dcc.Store(id='signal'),

    html.Div(id="initial_dummy_0"),  # Level zero initialization (Read available input parameter)
    html.Div(id="initial_dummy_1"),  # Level one initialization (e.g. creation of study table,...)

    # empty Div to trigger javascript file for graph resizing
    html.Div(id="output-clientside"),
    # modal for any warning
    dm.create_modal(),

    html.Div([  # HEADER (Header row, logo and title)
        html.Div(  # Logo
            html.Div(
                html.Img(
                    src=app.get_asset_url("logo-zbt.png"),
                    id="zbt-image",
                    style={"object-fit": 'contain',
                           'position': 'center',
                           "width": "auto",
                           "margin": "auto"}),
                id="logo-container", className="pretty_container h-100",
                style={'display': 'flex', 'justify-content': 'center',
                       'align-items': 'center'}
            ),
            className='col-12 col-lg-4 mb-2'
        ),
        html.Div(  # Title
            html.Div(
                html.H3("Fuel Cell Stack Model",
                        style={"margin": "auto",
                               "min-height": "47px",
                               "font-weight": "bold",
                               "-webkit-text-shadow-width": "1px",
                               "-webkit-text-shadow-color": "#aabad6",
                               "color": "#0062af",
                               "font-size": "40px",
                               "width": "auto",
                               "text-align": "center",
                               "vertical-align": "middle"}),
                className="pretty_container h-100", id="title",
                style={'justify-content': 'center', 'align-items': 'center',
                       'display': 'flex'}),
            style={'justify-content': 'space-evenly'},
            className='col-12 col-lg-8 mb-2'),
    ],
        id="header",
        className='row'
    ),

    html.Div([  # MIDDLE
        html.Div([  # LEFT MIDDLE / (Menu Column)
            # Menu Tabs
            html.Div([
                dl.tab_container(gui_input.main_frame_dicts)],
                id='setting_container'),  # style={'flex': '1'}
            # Buttons 1 (Load/Save Settings, Run
            html.Div([  # LEFT MIDDLE: Buttons
                html.Div([
                    html.Div([
                        dcc.Upload(id='upload-file',
                                   children=html.Button(
                                       'Load Settings',
                                       id='load-button',
                                       className='settings_button',
                                       style={'display': 'flex'})),
                        dcc.Download(id="savefile-json"),
                        html.Button('Save Settings', id='save-button',
                                    className='settings_button',
                                    style={'display': 'flex'}),
                        html.Button('Run single Simulation', id='run_button',
                                    className='settings_button',
                                    style={'display': 'flex'})
                    ],

                        style={'display': 'flex',
                               'flex-wrap': 'wrap',
                               # 'flex-direction': 'column',
                               # 'margin': '5px',
                               'justify-content': 'space-evenly'}
                    )],
                    className='neat-spacing')], style={'flex': '1'},
                id='load_save_run', className='pretty_container'),
            # Buttons 2 (UI Curve)
            html.Div([  # LEFT MIDDLE: Buttons
                html.Div([
                    html.Div([
                        html.Button('Calc. UI Curve', id='btn_init_ui',
                                    className='settings_button',
                                    style={'display': 'flex'}),
                        html.Button('refine ui', id='btn_refine_ui',
                                    className='settings_button',
                                    style={'display': 'flex'}),
                    ],

                        style={'display': 'flex',
                               'flex-wrap': 'wrap',
                               # 'flex-direction': 'column',
                               # 'margin': '5px',
                               'justify-content': 'space-evenly'}
                    )],
                    className='neat-spacing')], style={'flex': '1'},
                id='multiple_runs', className='pretty_container'),
            # Buttons 3 (Study)
            html.Div([

                html.Div(id="study_table"),
                html.Div(dcc.Checklist(id="check_calcUI", options=[{'label': 'Calc. complete Polarization Curve',
                                                                    'value': 'calcUI'}])),
                html.Div(dcc.RadioItems(id="check_studyType", options=[{'label': 'Single Variation', 'value': 'single'},
                                                                       {'label': 'Full Factorial', 'value': 'full'}],
                                        value='single',
                                        inline=True)),
                html.Div([
                    html.Div([
                        html.Button('study', id='btn_study',
                                    className='settings_button',
                                    style={'display': 'flex'}),

                    ],
                        style={'display': 'flex',
                               'flex-wrap': 'wrap',
                               'justify-content': 'space-evenly'}
                    )],
                    className='neat-spacing')], style={'flex': '1'},
                id='study', className='pretty_container'),

            # Buttons 4 (Save Results, Load Results, Update Plot (debug))
            html.Div([  # LEFT MIDDLE: Buttons
                html.Div([
                    html.Div([
                        html.Button('plot', id='btn_plot',
                                    className='settings_button',
                                    style={'display': 'flex'}
                                    ),
                        html.Button('SaveResults', id='btn_saveres',
                                    className='settings_button',
                                    style={'display': 'flex'}
                                    ),
                        dcc.Download(id="download-results"),

                        dcc.Upload(id='loadres', children=html.Button('Load Results', id='btn_loadres',
                                                                      className='settings_button',
                                                                      style={'display': 'flex'}
                                                                      ))
                    ],
                        style={'display': 'flex',
                               'flex-wrap': 'wrap',
                               'justify-content': 'space-evenly'}
                    )],
                    className='neat-spacing')], style={'flex': '1'},
                id='SaveLoad_Res', className='pretty_container'),
            html.Div([  # LEFT MIDDLE: Spinner
                html.Div([
                    html.Div([dbc.Spinner(html.Div(id="spinner_run_single")),
                              dbc.Spinner(html.Div(id="spinner_ui")),
                              dbc.Spinner(html.Div(id="spinner_uirefine")),
                              dbc.Spinner(html.Div(id="spinner_study")),
                              ],

                             # style={'display': 'flex',
                             #       'flex-wrap': 'wrap',
                             #       'justify-content': 'space-evenly'}
                             )],
                    className='neat-spacing')], style={'flex': '1'},
                id='spinner_bar', className='pretty_container'),
            # Progress Bar
            html.Div([
                # See: https://towardsdatascience.com/long-callbacks-in-dash-web-apps-72fd8de25937
                html.Div([
                    html.Div([pbar, timer_progress])],
                    className='neat-spacing')], style={'flex': '1'},
                id='progress_bar', className='pretty_container')
        ], id="left-column", className='col-12 col-lg-4 mb-2'),

        html.Div([  # RIGHT MIDDLE  (Result Column)
            html.Div(
                [html.Div('U-I-Curve', className='title'),
                 dcc.Graph(id='ui')],
                id='div_ui',
                className='pretty_container',
                style={'overflow': 'auto'}),

            html.Div([
                html.Div('Global Results (For Study only first dataset))', className='title'),
                dt.DataTable(id='global_data_table',
                             editable=True,
                             column_selectable='multi')],
                id='div_global_table',
                className='pretty_container',
                style={'overflow': 'auto'}),
            html.Div([
                html.Div('Heatmap', className='title'),
                html.Div(
                    [html.Div(
                        dcc.Dropdown(
                            id='dropdown_heatmap',
                            placeholder='Select Variable',
                            className='dropdown_input'),
                        id='div_results_dropdown',
                        # style={'padding': '1px', 'min-width': '200px'}
                    ),
                        html.Div(
                            dcc.Dropdown(id='dropdown_heatmap_2',
                                         className='dropdown_input',
                                         style={'visibility': 'hidden'}),
                            id='div_results_dropdown_2',
                        )
                    ],
                    style={'display': 'flex',
                           'flex-direction': 'row',
                           'flex-wrap': 'wrap',
                           'justify-content': 'left'},
                ),
                # RIGHT MIDDLE BOTTOM
                dbc.Spinner(dcc.Graph(id="heatmap_graph"),
                            spinner_class_name='loading_spinner',
                            fullscreen_class_name='loading_spinner_bg'),
            ],
                id='heatmap_container',
                className='graph pretty_container'),

            html.Div([
                html.Div('Plots', className='title'),
                html.Div(
                    [html.Div(
                        dcc.Dropdown(
                            id='dropdown_line',
                            placeholder='Select Variable',
                            className='dropdown_input'),
                        id='div_dropdown_line',
                        # style={'padding': '1px', 'min-width': '200px'}
                    ),
                        html.Div(
                            dcc.Dropdown(id='dropdown_line2',
                                         className='dropdown_input',
                                         style={'visibility': 'hidden'}),
                            id='div_dropdown_line_2',
                            # style={'padding': '1px', 'min-width': '200px'}
                        )],
                    style={'display': 'flex', 'flex-direction': 'row',
                           'flex-wrap': 'wrap',
                           'justify-content': 'left'},
                ),
                html.Div([
                    html.Div(
                        [
                            dcc.Store(id='append_check'),
                            html.Div(
                                [html.Div(
                                    children=dbc.DropdownMenu(
                                        id='checklist_dropdown',
                                        children=[
                                            dbc.Checklist(
                                                id='data_checklist',
                                                # input_checked_class_name='checkbox',
                                                style={
                                                    'max-height': '400px',
                                                    'overflow': 'auto'})],
                                        toggle_style={
                                            'textTransform': 'none',
                                            'background': '#fff',
                                            'border': '#ccc',
                                            'letter-spacing': '0',
                                            'font-size': '11px',
                                        },
                                        align_end=True,
                                        toggle_class_name='dropdown_input',
                                        label="Select Cells"), ),
                                    html.Button('Clear All', id='clear_all_button',
                                                className='local_data_buttons'),
                                    html.Button('Select All',
                                                id='select_all_button',
                                                className='local_data_buttons'),
                                    html.Button('Export to Table',
                                                id='export_b',
                                                className='local_data_buttons'),
                                    html.Button('Append to Table',
                                                id='append_b',
                                                className='local_data_buttons'),
                                    html.Button('Clear Table', id='clear_table_b',
                                                className='local_data_buttons')],
                                style={
                                    'display': 'flex',
                                    'flex-wrap': 'wrap',
                                    'margin-bottom': '5px'}
                            )],
                        # style={'width': '200px'}
                    ),
                    dcc.Store(id='cells_data')],
                    style={'display': 'flex', 'flex-direction': 'column',
                           'justify-content': 'left'}),
                dbc.Spinner(dcc.Graph(id='line_graph'),
                            spinner_class_name='loading_spinner',
                            fullscreen_class_name='loading_spinner_bg')],
                className="pretty_container",
                style={'display': 'flex', 'flex-direction':
                    'column', 'justify-content': 'space-evenly'}
            )],
            id='right-column', className='col-12 col-lg-8 mb-2')],
        className="row",
        style={'justify-content': 'space-evenly'}),

    # Data Table created from "Plots"
    html.Div(dt.DataTable(id='table', editable=True,
                          column_selectable='multi'),
             # columns=[{'filter_options': 'sensitive'}]),
             id='div_table', style={'overflow': 'auto',
                                    'position': 'relative',
                                    # 'margin': '0 0.05% 0 0.7%'
                                    },
             className='pretty_container'),

    # Bottom row, links to GitHub,...
    html.Div(
        html.Div([
            html.A('Source code:'),
            html.A('web interface',
                   href='https://www.github.com/zbt-tools/pemfc-dash-app',
                   target="_blank"),
            html.A("fuel cell model",
                   href='https://www.github.com/zbt-tools/pemfc-core',
                   target="_blank")],
            id='github_links',
            style={'overflow': 'auto',
                   'position': 'relative',
                   'justify-content': 'space-evenly',
                   'align-items': 'center',
                   'min-width': '30%',
                   'display': 'flex'}),
        id='link_container',
        style={'overflow': 'auto',
               'position': 'relative',
               'justify-content': 'center',
               'align-items': 'center',
               'display': 'flex'},
        className='pretty_container')
],
    id="mainContainer",
    # className='twelve columns',
    fluid=True,
    style={'padding': '0px'})


def variation_parameter(df_input: pd.DataFrame, keep_nominal=False, mode="single",
                        table_input=None) -> pd.DataFrame:
    """
    Function to create parameter sets.
    - variation of single parameter - ok
    - (single) variation of multiple parameters - ok
    - combined variation of multiple parameters
        - full factorial - ok

    Important: Change casting_func to int(),float(),... accordingly!
    """

    # Define parameter sets ( move to GUI)
    # -----------------------
    if not table_input:
        var_parameter = {
            "membrane-thickness": {"values": [0.25e-05, 4e-05], "casting": float},
            "cathode-electrochemistry-thickness_gdl": {"values": [0.00005, 0.0008], "casting": float},
            "anode-electrochemistry-thickness_gdl": {"values": [0.00005, 0.0008], "casting": float},
        }
    else:
        var_par_names = [le["Parameter"] for le in table_input if le["Parameter"] is not None]
        var_par_values = [ast.literal_eval(le["Value"]) for le in table_input if le["Parameter"] is not None]
        var_par_cast = [type(ast.literal_eval(le["Value"])[0]) for le in table_input if le["Parameter"] is not None]

        var_parameter = {name: {"values": val, "casting": cast} for name, val, cast in
                         zip(var_par_names, var_par_values, var_par_cast)}

    # Add informational column "variation_parameter"
    clms = list(df_input.columns)
    clms.extend(["variation_parameter"])
    data = pd.DataFrame(columns=clms)

    if mode == "single":  #
        # ... vary one variation_parameter, all other parameter nominal (from GUI)
        for parname, attr in var_parameter.items():
            for val in attr["values"]:
                inp = df_input.copy()
                inp.loc["nominal", parname] = attr["casting"](val)
                inp.loc["nominal", "variation_parameter"] = parname
                data = pd.concat([data, inp], ignore_index=True)

    elif mode == "full":
        # see https://docs.python.org/3/library/itertools.html

        parameter_names = [key for key, val in var_parameter.items()]
        parameter_names_string = ",".join(parameter_names)
        parameter_values = [val["values"] for key, val in var_parameter.items()]
        parameter_casting = [val["casting"] for key, val in var_parameter.items()]
        parameter_combinations = list(product(*parameter_values))

        for combination in parameter_combinations:
            inp = df_input.copy()
            inp.loc["nominal", "variation_parameter"] = parameter_names_string
            for par, val, cast in zip(parameter_names, combination, parameter_casting):
                inp.loc["nominal", par] = cast(val)
            data = pd.concat([data, inp], ignore_index=True)

    if keep_nominal:
        data = pd.concat([data, df_input])

    return data


def find_max_current_density(data: pd.DataFrame, df_input, settings):
    """
    Note: Expects one row DataFrame

    @param data:
    @param df_input:
    @param settings:
    @return:
    """
    # Find highest current density
    # ----------------------------
    success = False
    u_min = 0.05  # V

    data_backup = data.copy()

    while not success:
        data = data_backup.copy()
        u_min += 0.05

        # Change solver settings temporarily to voltage control
        data.loc[:, "simulation-operation_control"] = "Voltage"
        data.loc[:, "simulation-average_cell_voltage"] = u_min

        # Create complete setting dict, append it in additional column "settings" to df_input
        data = create_settings(data, settings, input_cols=df_input.columns)

        # Run simulation
        df_result, success = run_simulation(data)

    max_i = df_result["global_data"].iloc[0]["Average Current Density"]["value"]

    return max_i


def run_simulation(input_table: pd.DataFrame, return_unsuccessful=True) -> (pd.DataFrame, bool):
    """

    - Run input_table rows, catch exceptions of single calculations
            https://stackoverflow.com/questions/22847304/exception-handling-in-pandas-apply-function
    - Append result columns to input_table
    - Return DataFrame
    """

    def func(settings):
        try:
            return main_app.main(settings)
        except Exception as E:
            return repr(E)

    result_table = input_table["settings"].progress_apply(func)
    # result_table = input_table["settings"].map(func)
    # result_table = input_table["settings"].parallel_apply(func)
    # result_table = input_table["settings"].apply_parallel(func, num_processes=4)

    input_table["global_data"] = result_table.apply(lambda x: x[0][0] if (isinstance(x, tuple)) else None)
    input_table["local_data"] = result_table.apply(lambda x: x[1][0] if (isinstance(x, tuple)) else None)
    input_table["successful_run"] = result_table.apply(lambda x: True if (isinstance(x[0], list)) else False)

    all_successfull = True if input_table["successful_run"].all() else False

    if not return_unsuccessful:
        input_table = input_table.loc[input_table["successful_run"], :]

    return input_table, all_successfull


@app.callback(
    Output('pbar', 'value'),
    Output('pbar', 'label'),
    Output('pbar', 'color'),
    Input('timer_progress', 'n_intervals'),
    prevent_initial_call=True)
def cbf_progress_bar(*args) -> (float, str):
    """
    # https://towardsdatascience.com/long-callbacks-in-dash-web-apps-72fd8de25937
    """

    try:
        with open('progress.txt', 'r') as file:
            str_raw = file.read()
        last_line = list(filter(None, str_raw.split('\n')))[-1]
        percent = float(last_line.split('%')[0])

    except:
        percent = 0

    finally:
        text = f'{percent:.0f}%'
        if int(percent) == 100:
            color = "success"
        else:
            color = "primary"
        return percent, text, color


@app.callback(
    Output("pemfc_settings_file", "data"),
    Output('df_input_store', 'data'),
    Output("study_table", "children"),
    Input("initial_dummy_0", "children"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id')],
)
def cbf_initialization(dummy, inputs, inputs2, ids, ids2):
    """
    Initialization
    """
    # Read pemfc default settings.json file
    # --------------------------------------
    try:
        # Initially get default simulation settings from settings.json file
        # in pemfc core module
        pemfc_base_dir = os.path.dirname(pemfc.__file__)
        with open(os.path.join(pemfc_base_dir, 'settings', 'settings.json')) \
                as file:
            settings = json.load(file)
        # Avoid local outputs from simulation
        settings['output']['save_csv'] = False
        settings['output']['save_plot'] = False
    except Exception as E:
        print(repr(E))

    settings = df.store_data(settings)

    # Read initial data input
    # --------------------------------------
    # Read data from input fields and save input in dict/dataframe (one row "nominal")
    df_input = df.process_inputs(inputs, inputs2, ids, ids2, returntype="DataFrame")
    df_input_store = df.store_data(df_input)

    # Initialize study data table
    # -------------------------------------
    # Info: css stylesheet needed to be updated to show dropdown, see
    # https://community.plotly.com/t/resolved-dropdown-options-in-datatable-not-showing/20366/4

    # Dummy input
    empty_study_table = pd.DataFrame(dict([
        ('Parameter', ["membrane-thickness", None, None, None]),
        ('Value', ["[1e-5, 2e-5]", None, None, None]),
        # ('ValueType', ["float", None, None, None])
    ]))

    table = dash_table.DataTable(
        id='table-dropdown',
        data=empty_study_table.to_dict('records'),
        columns=[
            {'id': 'Parameter', 'name': 'Parameter', 'presentation': 'dropdown'},
            {'id': 'Value', 'name': 'Values'},
            # {'id': 'ValueType', 'name': 'Value Type', 'presentation': 'dropdown'},

        ],

        editable=True,
        dropdown={
            'Parameter': {
                'options': [
                    {'label': i, 'value': i}
                    for i in list(df_input.columns)
                ]},
            # 'ValueType': {
            #     'options': [
            #         {'label': i, 'value': i}
            #         for i in ["int", "float"]
            #     ]}
        }
    )

    print("init successful")
    return settings, df_input_store, table


# @app.callback(
#     Output("study_table", "children"),
#     Input("initial_dummy_1", "children"),
#     State('df_input_store', 'data'),
#     prevent_initial_call=True)
# def cbf_initial_study_table(inp, state):
#     print("yo")
#     df_nominal = df.read_data(ctx.states["df_input_store.data"])
#     empty_study_table = pd.DataFrame(dict([
#         ('Parameter', [None, None, None, None]),
#         ('Value', [None, None, None, None])]))
#
#     table = dash_table.DataTable(
#         id='table-dropdown',
#         data=empty_study_table.to_dict('records'),
#         columns=[
#             {'id': 'Parameter', 'name': 'Parameter', 'presentation': 'dropdown'},
#             {'id': 'Value', 'name': 'Values'},
#         ],
#
#         editable=True,
#         dropdown={
#             'Parameter': {
#                 'options': [
#                     {'label': i, 'value': i}
#                     for i in df_nominal.columns
#                 ]}}
#     )
#
#     return table


@app.callback(
    Output('df_result_data_store', 'data'),
    Output('df_input_store', 'data'),
    Output('signal', 'data'),
    Output("spinner_run_single", 'children'),
    Input("run_button", "n_clicks"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id'),
     State("pemfc_settings_file", "data")],
    prevent_initial_call=True)
def cbf_run_single_cal(n_click, inputs, inputs2, ids, ids2, settings):
    """
    Changelog:

    """
    try:
        # Read pemfc settings.json from store
        settings = df.read_data(settings)

        # Read data from input fields and save input in dict/dataframe (one row "nominal")
        df_input = df.process_inputs(inputs, inputs2, ids, ids2, returntype="DataFrame")
        df_input_raw = df_input.copy()

        # Create complete setting dict, append it in additional column "settings" to df_input
        df_input = create_settings(df_input, settings)

        # Run simulation
        df_result, _ = run_simulation(df_input)

        # Save results
        df_result_store = df.store_data(df_result)
        df_input_store = df.store_data(df_input_raw)

        return df_result_store, df_input_store, n_click, ""

    except Exception as E:
        modal_title, modal_body = \
            dm.modal_process('input-error', error=repr(E))


@app.callback(
    Output('df_result_data_store', 'data'),
    Output('df_input_store', 'data'),
    Output('spinner_ui', 'children'),
    Input("btn_init_ui", "n_clicks"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id'),
     State("pemfc_settings_file", "data")],
    prevent_initial_call=True)
def cbf_run_initial_ui_calculation(btn, inputs, inputs2, ids, ids2, settings):
    """

    #ToDO: 
        - Error handling: Handle None in results.

    @param btn: 
    @param inputs: 
    @param inputs2: 
    @param ids: 
    @param ids2: 
    @param settings:
    @return: 
    """

    n_refinements = 10

    # Progress bar init
    std_err_backup = sys.stderr
    file_prog = open('progress.txt', 'w')
    sys.stderr = file_prog

    # Read pemfc settings.json from store
    settings = df.read_data(settings)

    # Read data from input fields and save input in dict/dataframe (one row "nominal")
    df_input = df.process_inputs(inputs, inputs2, ids, ids2, returntype="DataFrame")
    df_input_backup = df_input.copy()

    # Ensure DataFrame with double bracket
    # https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
    df_input_single = df_input.loc[["nominal"], :]
    max_i = find_max_current_density(df_input_single, df_input, settings)

    # Reset solver settings
    df_input = df_input_backup.copy()

    # Prepare & calculate initial points
    df_results = uicalc_prepare_initcalc(input_df=df_input, i_limits=[1, max_i], settings=settings)
    df_results, success = run_simulation(df_results)

    # First refinement steps
    for _ in range(n_refinements):
        df_refine = uicalc_prepare_refinement(data_df=df_results, input_df=df_input, settings=settings)
        df_refine, success = run_simulation(df_refine, return_unsuccessful=False)
        df_results = pd.concat([df_results, df_refine], ignore_index=True)

    # Save results
    results = df.store_data(df_results)
    df_input_store = df.store_data(df_input_backup)

    # Close process bar files
    file_prog.close()
    sys.stderr = std_err_backup
    return results, df_input_store, "."


@app.callback(
    Output('df_result_data_store', 'data'),
    Output('spinner_uirefine', 'children'),
    Input("btn_refine_ui", "n_clicks"),
    State('df_result_data_store', 'data'),
    State('df_input_store', 'data'),
    State("pemfc_settings_file", "data"),
    prevent_initial_call=True)
def cbf_run_refine_ui(inp, state, state2, settings):
    # Number of refinement steps
    n_refinements = 5

    # Progress bar init
    std_err_backup = sys.stderr
    file_prog = open('progress.txt', 'w')
    sys.stderr = file_prog

    # Read pemfc settings.json from store
    settings = df.read_data(settings)

    # State-Store access returns None, I don't know why (FKL), workaround:
    df_results = df.read_data(ctx.states["df_result_data_store.data"])
    df_nominal = df.read_data(ctx.states["df_input_store.data"])

    # Refinement loop
    for _ in range(n_refinements):
        df_refine = uicalc_prepare_refinement(data_df=df_results, input_df=df_nominal, settings=settings)
        df_refine, success = run_simulation(df_refine, return_unsuccessful=False)
        df_results = pd.concat([df_results, df_refine], ignore_index=True)

    # Save results
    results = df.store_data(df_results)

    # Close process bar files
    file_prog.close()
    sys.stderr = std_err_backup
    return results, ""


@app.callback(
    Output('df_result_data_store', 'data'),
    Output('df_input_store', 'data'),
    Output('spinner_study', 'children'),
    Input("btn_study", "n_clicks"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id')],
    State("pemfc_settings_file", "data"),
    State("table-dropdown", "data"),
    State("check_calcUI", "value"),
    State("check_studyType", "value"),
    prevent_initial_call=True)
def cbf_run_study(btn, inputs, inputs2, ids, ids2, settings, tabledata, checkCalcUI, checkStudyType):
    """
    #ToDO Documentation
    """
    # Calculation of polarization curve for each dataset?
    if isinstance(checkCalcUI, list):
        ui_calculation = True
    else:
        ui_calculation = False
    n_refinements = 10

    mode = checkStudyType

    # Progress bar init
    std_err_backup = sys.stderr
    file_prog = open('progress.txt', 'w')
    sys.stderr = file_prog

    # Read pemfc settings.json from store
    settings = df.read_data(settings)

    # Read data from input fields and save input in dict (legacy) / pd.DataDrame (one row with index "nominal")
    df_input = df.process_inputs(inputs, inputs2, ids, ids2, returntype="DataFrame")
    df_input_backup = df_input.copy()

    # Create multiple parameter sets
    data = variation_parameter(df_input, keep_nominal=False, mode=mode, table_input=tabledata)
    varpars = list(data["variation_parameter"].unique())

    if not ui_calculation:
        # Create complete setting dict & append it in additional column "settings" to df_input
        data = create_settings(data, settings, input_cols=df_input.columns)
        # Run Simulation
        results, success = run_simulation(data)
        results = df.store_data(results)

    else:  # ... calculate pol. curve for each parameter set
        result_data = pd.DataFrame(columns=data.columns)

        # grouped_data = data.groupby(varpars, sort=False)
        # for _, group in grouped_data:
        for i in range(0, len(data)):

            print(f"Group: {i},start")
            # Ensure DataFrame with double bracket
            # https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
            # df_input_single = df_input.loc[[:], :]
            print(f"Group: {i}, Calc maxi")
            max_i = find_max_current_density(data.iloc[[i]], df_input, settings)

            # # Reset solver settings
            # df_input = df_input_backup.copy()

            success = False
            while (not success) and (max_i > 5000):
                print(f"Group: {i}, prep init ui, max_i:{max_i}")
                # Prepare & calculate initial points
                df_results = uicalc_prepare_initcalc(input_df=data.iloc[[i]], i_limits=[1, max_i], settings=settings,
                                                     input_cols=df_input.columns)
                print(f"Group: {i}, Calc init ui")
                df_results, success = run_simulation(df_results)
                max_i -= 2000

            if not success:
                continue

                # First refinement steps
            print(f"Group: {i}, prep refine")
            for _ in range(n_refinements):
                print(f"Refinement itenration {_}")
                df_refine = uicalc_prepare_refinement(input_df=df_input, data_df=df_results, settings=settings)
                df_refine, success = run_simulation(df_refine, return_unsuccessful=False)
                df_results = pd.concat([df_results, df_refine], ignore_index=True)

            result_data = pd.concat([result_data, df_results], ignore_index=True)
            print(f"Group: {i}, finish")

            gc.collect()

        results = df.store_data(result_data)

    df_input_store = df.store_data(df_input_backup)

    file_prog.close()
    sys.stderr = std_err_backup

    return results, df_input_store, "."


@app.callback(
    Output("download-results", "data"),
    Input("btn_saveres", "n_clicks"),
    State('df_result_data_store', 'data'),
    prevent_initial_call=True)
def cbf_save_results(inp, state):
    # State-Store access returns None, I don't know why (FKL)
    data = ctx.states["df_result_data_store.data"]
    with open('results.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dcc.send_file("results.pickle")


@app.callback(
    Output('df_result_data_store', 'data'),
    Input("loadres", "contents"),
    prevent_initial_call=True)
def cbf_load_results(content):
    # https://dash.plotly.com/dash-core-components/upload
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    b = pickle.load(io.BytesIO(decoded))

    return b


@app.callback(
    Output('ui', 'figure'),
    Input('df_result_data_store', 'data'),
    Input('btn_plot', 'n_clicks'),
    State('df_input_store', 'data'),
    prevent_initial_call=True)
def cbf_figure_ui(inp1, inp2, dfinp):
    """
    Prior to plot: identification of same parameter sets with different current density.
    Those points will be connected and have identical color
    """

    # Read results
    results = df.read_data(ctx.inputs["df_result_data_store.data"])
    df_nominal = df.read_data(ctx.states["df_input_store.data"])
    results = results.loc[results["successful_run"] == True, :]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.data = []

    # Check for identical parameter, only different current density
    group_columns = list(df_nominal.columns)
    group_columns.remove('simulation-current_density')
    # Groupby fails, as data contains lists, which are not hashable, therefore conversion to tuple
    # see https://stackoverflow.com/questions/52225301/error-unhashable-type-list-while-using-df-groupby-apply
    # see https://stackoverflow.com/questions/51052416/pandas-dataframe-groupby-into-list-with-list-in-cell-data
    # results_red = results.loc[:, df_nominal.columns].copy()
    results = results.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
    grouped = results.groupby(group_columns, sort=False)

    for _, group in grouped:
        group.sort_values("simulation-current_density", ignore_index=True, inplace=True)
        group["Voltage"] = group["global_data"].apply(
            lambda x: x["Stack Voltage"]["value"] if (x is not None) else None)
        group["Power"] = group["global_data"].apply(lambda x: x["Stack Power"]["value"] if (x is not None) else None)

        # Add traces
        if "variation_parameter" in group.columns:
            # Variation parameter can be one parameter or multiple parameter separated by ",".
            varpar = group["variation_parameter"][0]
            try:
                if varpar.find(',') == -1:  # no parameter separator -> one parameter:
                    setname = f"{varpar}: {group[varpar][0]}"
                    # Add figure title
                    fig.update_layout(
                        title_text=f"U-i-Curve"
                    )
                else:
                    setname = f"{', '.join([f'par{n}: {group[vp][0]}' for n, vp in enumerate(varpar.split(','))])}"
                    # Add figure title
                    fig.update_layout(
                        title_text=f"U-i-Curve, Variation parameter: <br> {[par for par in varpar.split(',')]}"
                    )

            except:
                setname = "tbd"
                # Add figure title
                fig.update_layout(
                    title_text=f"U-i-Curve")

        else:
            # Add figure title
            fig.update_layout(
                title_text=f"U-i-Curve"
            )
            setname = ""

        fig.add_trace(
            go.Scatter(x=group["simulation-current_density"], y=group["Voltage"], name=f"{setname},  U [V]",
                       mode='lines+markers'),
            secondary_y=False,
        )

        # fig.add_trace(
        #    go.Scatter(x=group["simulation-current_density"], y=group["Power"], name=f"{setname}, Power",
        #               mode='lines+markers'),
        #    secondary_y=True,
        # )

    # Set x-axis title
    fig.update_xaxes(title_text="i [A/m²]")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Voltage</b> U [V]", secondary_y=False)
    fig.update_yaxes(title_text="<b>Power</b> P [W]", secondary_y=True)
    fig.update_layout(hoverlabel=dict(namelength=-1))

    return fig


@app.callback(
    [Output('global_data_table', 'columns'),
     Output('global_data_table', 'data'),
     Output('global_data_table', 'export_format')],
    Input('df_result_data_store', 'data'),
    prevent_initial_call=True
)
def global_outputs_table(*args):
    """
    ToDo: Add additional input.
    If storage triggered callback, use first result row,
    if dropdown triggered callback, select this row.
    """

    # Read results
    results = df.read_data(ctx.inputs["df_result_data_store.data"])

    result_set = results.iloc[0]

    global_result_dict = result_set["global_data"]
    names = list(global_result_dict.keys())
    values = [v['value'] for k, v in global_result_dict.items()]
    units = [v['units'] for k, v in global_result_dict.items()]

    column_names = ['Quantity', 'Value', 'Units']
    columns = [{'deletable': True, 'renamable': True,
                'selectable': True, 'name': col, 'id': col}
               for col in column_names]
    datas = [{column_names[0]: names[i],
              column_names[1]: values[i],
              column_names[2]: units[i]} for i in range(len(values))]

    return columns, datas, 'csv',


@app.callback(
    [Output('dropdown_heatmap', 'options'),
     Output('dropdown_heatmap', 'value')],
    Input('df_result_data_store', 'data'),
    prevent_initial_call=True
)
def get_dropdown_options_heatmap(results):
    """
    ToDo: Add additional input.
    If storage triggered callback, use first result row,
    if dropdown triggered callback, select this row.
    """

    # Read results
    results = df.read_data(ctx.inputs["df_result_data_store.data"])

    result_set = results.iloc[0]

    local_data = result_set["local_data"]
    values = [{'label': key, 'value': key} for key in local_data
              if 'xkey' in local_data[key]
              and local_data[key]['xkey'] == 'Channel Location']
    return values, 'Current Density'


@app.callback(
    [Output('dropdown_line', 'options'),
     Output('dropdown_line', 'value')],
    Input('df_result_data_store', 'data'),
    prevent_initial_call=True
)
def get_dropdown_options_line_graph(results):
    """
    ToDo: Add additional input.
    If storage triggered callback, use first result row,
    if dropdown triggered callback, select this row.
    """

    # Read results
    results = df.read_data(ctx.inputs["df_result_data_store.data"])

    result_set = results.iloc[0]

    local_data = result_set["local_data"]
    values = [{'label': key, 'value': key} for key in local_data]
    return values, 'Current Density'


@app.callback(
    [Output('dropdown_heatmap_2', 'options'),
     Output('dropdown_heatmap_2', 'value'),
     Output('dropdown_heatmap_2', 'style')],
    [Input('dropdown_heatmap', 'value'),
     Input('df_result_data_store', 'data')]
)
def get_dropdown_options_heatmap_2(dropdown_key, results):
    if dropdown_key is None or results is None:
        raise PreventUpdate
    else:
        # Read results
        results = df.read_data(ctx.inputs["df_result_data_store.data"])

        result_set = results.iloc[0]
        local_data = result_set["local_data"]
        if 'value' in local_data[dropdown_key]:
            return [], None, {'visibility': 'hidden'}
        else:
            options = [{'label': key, 'value': key} for key in
                       local_data[dropdown_key]]
            value = options[0]['value']
            return options, value, {'visibility': 'visible'}


@app.callback(
    [Output('dropdown_line2', 'options'),
     Output('dropdown_line2', 'value'),
     Output('dropdown_line2', 'style')],
    Input('dropdown_line', 'value'),
    Input('df_result_data_store', 'data'),
    prevent_initial_call=True
)
def get_dropdown_options_line_graph_2(dropdown_key, results):
    if dropdown_key is None or results is None:
        raise PreventUpdate
    else:
        # Read results
        results = df.read_data(ctx.inputs["df_result_data_store.data"])

        result_set = results.iloc[0]

        local_data = result_set["local_data"]
        if 'value' in local_data[dropdown_key]:
            return [], None, {'visibility': 'hidden'}
        else:
            options = [{'label': key, 'value': key} for key in
                       local_data[dropdown_key]]
            value = options[0]['value']
            return options, value, {'visibility': 'visible'}


@app.callback(
    Output("heatmap_graph", "figure"),
    [Input('dropdown_heatmap', 'value'),
     Input('dropdown_heatmap_2', 'value')],
    Input('df_result_data_store', 'data'),
    prevent_initial_call=True
)
def update_heatmap_graph(dropdown_key, dropdown_key_2, results):
    if dropdown_key is None or results is None:
        raise PreventUpdate
    else:
        # Read results
        results = df.read_data(ctx.inputs["df_result_data_store.data"])

        result_set = results.iloc[0]

        local_data = result_set["local_data"]

        if 'value' in local_data[dropdown_key]:
            zvalues = local_data[dropdown_key]['value']
        elif dropdown_key_2 is not None:
            zvalues = local_data[dropdown_key][dropdown_key_2]['value']
        else:
            raise PreventUpdate

        x_key = local_data[dropdown_key]['xkey']
        y_key = 'Cells'
        xvalues = np.asarray(local_data[x_key]['value'])
        if xvalues.ndim > 1:
            xvalues = xvalues[0]
        yvalues = np.asarray(local_data[y_key]['value'])
        if yvalues.ndim > 1:
            yvalues = yvalues[0]

        n_y = len(yvalues)
        n_x = xvalues.shape[-1]
        n_z = yvalues.shape[-1]

        if n_x == n_z + 1:
            xvalues = ip.interpolate_1d(xvalues)

        if dropdown_key_2 is None:
            z_title = dropdown_key + ' / ' + local_data[dropdown_key]['units']
        else:
            z_title = dropdown_key + ' / ' \
                      + local_data[dropdown_key][dropdown_key_2]['units']

        # if n_y <= 20:
        #     height = 300
        # elif 20 < n_y <= 100:
        #     height = 300 + n_y * 10.0
        # else:
        #     height = 1300

        height = 800
        # width = 500

        font_props = dl.graph_font_props

        base_axis_dict = \
            {'tickfont': font_props['medium'],
             'titlefont': font_props['large'],
             'title': x_key + ' / ' + local_data[x_key]['units'],
             'tickmode': 'array', 'showgrid': True}

        tick_division_dict = \
            {'fine': {'upper_limit': 10, 'value': 1},
             'medium': {'upper_limit': 20, 'value': 2},
             'medium_coarse': {'upper_limit': 50, 'value': 5},
             'coarse': {'value': 10}}

        def filter_tick_text(data, spacing=1):
            return [str(data[i]) if i % spacing == 0 else ' '
                    for i in range(len(data))]

        def granular_tick_division(data, division=None):
            n = len(data)
            if division is None:
                division = tick_division_dict
            if n <= division['fine']['upper_limit']:
                result = filter_tick_text(data, division['fine']['value'])
            elif division['fine']['upper_limit'] < n \
                    <= division['medium']['upper_limit']:
                result = \
                    filter_tick_text(data, division['medium']['value'])
            elif division['medium']['upper_limit'] < n \
                    <= division['medium_coarse']['upper_limit']:
                result = filter_tick_text(
                    data, division['medium_coarse']['value'])
            else:
                result = \
                    filter_tick_text(data, division['coarse']['value'])
            return result

        # y_tick_labels[-1] = str(n_y - 1)

        x_axis_dict = copy.deepcopy(base_axis_dict)
        x_axis_dict['title'] = x_key + ' / ' + local_data[x_key]['units']
        x_axis_dict['tickvals'] = local_data[x_key]['value']
        x_axis_dict['ticktext'] = \
            granular_tick_division(local_data[x_key]['value'])

        y_axis_dict = copy.deepcopy(base_axis_dict)
        y_axis_dict['title'] = y_key + ' / ' + local_data[y_key]['units']
        y_axis_dict['tickvals'] = yvalues
        y_axis_dict['ticktext'] = granular_tick_division(range(n_y))

        z_axis_dict = copy.deepcopy(base_axis_dict)
        z_axis_dict['title'] = z_title
        # z_axis_dict['tickvals'] = zvalues

        layout = go.Layout(
            font=font_props['large'],
            # title='Local Results in Heat Map',
            titlefont=font_props['large'],
            xaxis=x_axis_dict,
            yaxis=y_axis_dict,
            margin={'l': 75, 'r': 20, 't': 10, 'b': 20},
            height=height
        )
        scene = dict(
            xaxis=x_axis_dict,
            yaxis=y_axis_dict,
            zaxis=z_axis_dict)

        heatmap = \
            go.Surface(z=zvalues, x=xvalues, y=yvalues,  # xgap=1, ygap=1,
                       colorbar={
                           'tickfont': font_props['large'],
                           'title': {
                               'text': z_title,
                               'font': {'size': font_props['large']['size']},
                               'side': 'right'},
                           # 'height': height - 300
                           'lenmode': 'fraction',
                           'len': 0.75
                       })

        fig = go.Figure(data=heatmap, layout=layout)
        fig.update_layout(scene=scene)

    return fig


@app.callback(
    [Output('line_graph', 'figure'),
     Output('cells_data', 'data'),
     Output('data_checklist', 'options'),
     Output('data_checklist', 'value')],
    [Input('dropdown_line', 'value'),
     Input('dropdown_line2', 'value'),
     Input('data_checklist', 'value'),
     Input('select_all_button', 'n_clicks'),
     Input('clear_all_button', 'n_clicks'),
     Input('line_graph', 'restyleData')],
    Input('df_result_data_store', 'data'),
    prevent_initial_call=True
)
def update_line_graph(drop1, drop2, checklist, select_all_clicks,
                      clear_all_clicks, restyle_data, results):
    ctx_triggered = dash.callback_context.triggered[0]['prop_id']
    if drop1 is None or results is None:
        raise PreventUpdate
    else:
        # Read results
        results = df.read_data(ctx.inputs["df_result_data_store.data"])

        result_set = results.iloc[0]

        local_data = result_set["local_data"]

        fig = go.Figure()

        default_x_key = 'Number'
        x_key = local_data[drop1].get('xkey', default_x_key)

        if drop2 is None:
            y_title = drop1 + ' / ' + local_data[drop1]['units']
        else:
            y_title = drop1 + ' - ' + drop2 + ' / ' \
                      + local_data[drop1][drop2]['units']

        if x_key == default_x_key:
            x_title = x_key + ' / -'
        else:
            x_title = x_key + ' / ' + local_data[x_key]['units']

        if 'Error' in y_title:
            y_scale = 'log'
        else:
            y_scale = 'linear'

        layout = go.Layout(
            font={'color': 'black', 'family': 'Arial'},
            # title='Local Results in Heat Map',
            titlefont={'size': 11, 'color': 'black'},
            xaxis={'tickfont': {'size': 11}, 'titlefont': {'size': 14},
                   'title': x_title},
            yaxis={'tickfont': {'size': 11}, 'titlefont': {'size': 14},
                   'title': y_title},
            margin={'l': 100, 'r': 20, 't': 20, 'b': 20},
            yaxis_type=y_scale)

        fig.update_layout(layout)

        if 'value' in local_data[drop1]:
            yvalues = np.asarray(local_data[drop1]['value'])
        elif drop2 is not None:
            yvalues = np.asarray(local_data[drop1][drop2]['value'])
        else:
            raise PreventUpdate

        n_y = np.asarray(yvalues).shape[0]
        if x_key in local_data:
            xvalues = np.asarray(local_data[x_key]['value'])
            if len(xvalues) == n_y + 1:
                xvalues = ip.interpolate_1d(xvalues)
        else:
            xvalues = np.asarray(list(range(n_y)))

        if xvalues.ndim > 1:
            xvalues = xvalues[0]

        if yvalues.ndim == 1:
            yvalues = [yvalues]
        cells = {}
        for num, yval in enumerate(yvalues):
            fig.add_trace(go.Scatter(x=xvalues, y=yval,
                                     mode='lines+markers',
                                     name='Cell {}'.format(num)))
            cells[num] = {'name': 'Cell {}'.format(num), 'data': yval}

        options = [{'label': cells[k]['name'], 'value': cells[k]['name']}
                   for k in cells]
        value = ['Cell {}'.format(str(i)) for i in range(n_y)]

        if checklist is None:
            return fig, cells, options, value
        else:
            if 'clear_all_button.n_clicks' in ctx_triggered:
                fig.for_each_trace(
                    lambda trace: trace.update(visible='legendonly'))
                return fig, cells, options, []
            elif 'data_checklist.value' in ctx_triggered:
                fig.for_each_trace(
                    lambda trace: trace.update(
                        visible=True) if trace.name in checklist
                    else trace.update(visible='legendonly'))
                return fig, cells, options, checklist
            elif 'line_graph.restyleData' in ctx_triggered:
                read = restyle_data[0]['visible']
                if len(read) == 1:
                    cell_name = cells[restyle_data[1][0]]['name']
                    if read[0] is True:  # lose (legendonly)
                        checklist.append(cell_name)
                    else:
                        if cell_name in checklist:
                            checklist.remove(cell_name)
                    value = [val for val in value if val in checklist]
                else:
                    value = [value[i] for i in range(n_y)
                             if read[i] is True]
                fig.for_each_trace(
                    lambda trace: trace.update(
                        visible=True) if trace.name in value
                    else trace.update(visible='legendonly'))
                # fig.plotly_restyle(restyle_data[0])
                return fig, cells, options, value
            else:
                return fig, cells, options, value


@app.callback(
    [Output('table', 'columns'),
     Output('table', 'data'),
     Output('table', 'export_format'),
     Output('append_check', 'data')],
    [Input('export_b', 'n_clicks'),  # button1
     Input('append_b', 'n_clicks'),  # button2
     Input('clear_table_b', 'n_clicks')],
    [State('data_checklist', 'value'),  # from display
     State('cells_data', 'data'),  # from line graph
     State('df_result_data_store', 'data'),
     State('table', 'columns'),
     State('table', 'data'),
     State('append_check', 'data')],
    prevent_initial_call=True
)
def list_to_table(n1, n2, n3, data_checklist, cells_data, results,
                  table_columns, table_data, append_check):
    ctx = dash.callback_context.triggered[0]['prop_id']
    if data_checklist is None or results is None:
        raise PreventUpdate
    else:
        # Read results
        results = df.read_data(ctx.inputs["df_result_data_store.data"])

        result_set = results.iloc[0]

        local_data = result_set["local_data"]
        digit_list = \
            sorted([int(re.sub('[^0-9\.]', '', inside))
                    for inside in data_checklist])

        x_key = 'Channel Location'

        index = [{'id': x_key, 'name': x_key,
                  'deletable': True}]
        columns = [{'deletable': True, 'renamable': True,
                    'selectable': True, 'name': 'Cell {}'.format(d),
                    'id': 'Cell {}'.format(d)} for d in digit_list]
        # list with nested dict

        xvalues = ip.interpolate_1d(np.asarray(local_data[x_key]['value']))
        data = [{**{x_key: cell},
                 **{cells_data[k]['name']: cells_data[k]['data'][num]
                    for k in cells_data}}
                for num, cell in enumerate(xvalues)]  # list with nested dict

        if append_check is None:
            appended = 0
        else:
            appended = append_check

        if 'export_b.n_clicks' in ctx:
            return index + columns, data, 'csv', appended
        elif 'clear_table_b.n_clicks' in ctx:
            return [], [], 'none', appended
        elif 'append_b.n_clicks' in ctx:
            if n1 is None or table_data == [] or table_columns == [] or \
                    ctx == 'clear_table_b.n_clicks':
                raise PreventUpdate
            else:
                appended += 1
                app_columns = \
                    [{'deletable': True, 'renamable': True,
                      'selectable': True, 'name': 'Cell {}'.format(d),
                      'id': 'Cell {}'.format(d) + '-' + str(appended)}
                     for d in digit_list]
                new_columns = table_columns + app_columns
                app_datas = \
                    [{**{x_key: cell},
                      **{cells_data[k]['name'] + '-' + str(appended):
                             cells_data[k]['data'][num]
                         for k in cells_data}}
                     for num, cell in enumerate(xvalues)]
                new_data_list = []
                new_data = \
                    [{**table_data[i], **app_datas[i]}
                     if table_data[i][x_key] == app_datas[i][x_key]
                     else new_data_list.extend([table_data[i], app_datas[i]])
                     for i in range(len(app_datas))]
                new_data = list(filter(None.__ne__, new_data))
                new_data.extend(new_data_list)

                return new_columns, new_data, 'csv', appended
        else:
            if n1 is None or table_columns == []:
                return table_columns, table_data, 'none', appended
            else:
                return table_columns, table_data, 'csv', appended


@app.callback(
    [Output({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     Output({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     Output('upload-file', 'contents'),
     Output('modal-title', 'children'),
     Output('modal-body', 'children'),
     Output('modal', 'is_open')],
    Input('upload-file', 'contents'),
    [State('upload-file', 'filename'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id'),
     State('modal', 'is_open')]
)
def load_settings(contents, filename, value, multival, ids, ids2,
                  modal_state):
    if contents is None:
        raise PreventUpdate
    else:
        if 'json' in filename:
            try:
                j_file, err_l = df.parse_contents(contents)

                dict_ids = {id_l: val for id_l, val in
                            zip([id_l['id'] for id_l in ids], value)}
                dict_ids2 = {id_l: val for id_l, val in
                             zip([id_l['id'] for id_l in ids2], multival)}

                id_match = set.union(set(dict_ids),
                                     set([item[:-2] for item in dict_ids2]))

                for k, v in j_file.items():
                    if k in id_match:
                        if isinstance(v, list):
                            for num, val in enumerate(v):
                                dict_ids2[k + f'_{num}'] = df.check_ifbool(val)
                        else:
                            dict_ids[k] = df.check_ifbool(v)
                    else:
                        continue

                if not err_l:
                    # All JSON settings match Dash IDs
                    modal_title, modal_body = dm.modal_process('loaded')
                    return list(dict_ids.values()), list(dict_ids2.values()), \
                        None, modal_title, modal_body, not modal_state
                else:
                    # Some JSON settings do not match Dash IDs; return values
                    # that matched with Dash IDs
                    modal_title, modal_body = \
                        dm.modal_process('id-not-loaded', err_l)
                    return list(dict_ids.values()), list(dict_ids2.values()), \
                        None, modal_title, modal_body, not modal_state
            except Exception as E:
                # Error / JSON file cannot be processed; return old value
                modal_title, modal_body = \
                    dm.modal_process('error', error=repr(E))
                return value, multival, None, modal_title, modal_body, \
                    not modal_state
        else:
            # Not JSON file; return old value
            modal_title, modal_body = dm.modal_process('wrong-file')
            return value, multival, None, modal_title, modal_body, \
                not modal_state


@app.callback(
    [Output("savefile-json", "data")],
    Input('save-button', "n_clicks"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id')],
    prevent_initial_call=True,
)
def save_settings(n_clicks, val1, val2, ids, ids2):
    """

    @param n_clicks:
    @param val1:
    @param val2:
    @param ids:
    @param ids2:
    @return:
    """
    save_complete = True

    dict_data = df.process_inputs(val1, val2, ids, ids2)  # values first

    if not save_complete:  # ... save only GUI inputs
        sep_id_list = [joined_id.split('-') for joined_id in
                       dict_data.keys()]

        val_list = dict_data.values()
        new_dict = {}
        for sep_id, vals in zip(sep_id_list, val_list):
            current_level = new_dict
            for id_l in sep_id:
                if id_l not in current_level:
                    if id_l != sep_id[-1]:
                        current_level[id_l] = {}
                    else:
                        current_level[id_l] = vals
                current_level = current_level[id_l]

        return dict(content=json.dumps(new_dict, sort_keys=True, indent=2),
                    filename='settings.json')

    else:  # ... save complete settings as passed to pemfc simulation

        # code portion of generate_inputs()
        # ------------------------
        input_data = {}
        for k, v in dict_data.items():
            input_data[k] = {'sim_name': k.split('-'), 'value': v}

        # code portion of run_simulation()
        # ------------------------

        pemfc_base_dir = os.path.dirname(pemfc.__file__)
        with open(os.path.join(pemfc_base_dir, 'settings', 'settings.json')) \
                as file:
            settings = json.load(file)
        settings, _ = data_transfer.gui_to_sim_transfer(input_data, settings)

        return dict(content=json.dumps(settings, indent=2),
                    filename='settings.json')


@app.callback(
    Output({'type': ALL, 'id': ALL, 'specifier': 'disable_basewidth'},
           'disabled'),
    Input({'type': ALL, 'id': ALL, 'specifier': 'dropdown_activate_basewidth'},
          'value'))
def tab1_callback_disabled(value):
    for num, val in enumerate(value):
        if val == "trapezoidal":
            value[num] = False
        else:
            value[num] = True
    return value


@app.callback(
    Output({'type': ALL, 'id': ALL, 'specifier': 'disabled_manifolds'},
           'disabled'),
    Input({'type': ALL, 'id': ALL,
           'specifier': 'checklist_activate_calculation'}, 'value'),
    Input({'type': ALL, 'id': ALL, 'specifier': 'disabled_manifolds'}, 'value'),

)
def tab2_callback_activate_column(input1, input2):
    len_state = len(input2)
    list_state = [True for x in range(len_state)]  # disable=True for all inputs
    for num, val in enumerate(input1):  # 3 inputs in input1 for 3 rows
        if val == [1]:
            list_state[0 + num] = list_state[3 + num] = list_state[15 + num] = \
                list_state[18 + num] = list_state[30 + num] = False
            if input2[3 + num] == 'circular':
                list_state[6 + num], list_state[9 + num], list_state[12 + num] = \
                    False, True, True
            else:
                list_state[6 + num], list_state[9 + num], list_state[12 + num] = \
                    True, False, False
            if input2[18 + num] == 'circular':
                list_state[21 + num], list_state[24 + num], list_state[27 + num] = \
                    False, True, True
            else:
                list_state[21 + num], list_state[24 + num], list_state[27 + num] = \
                    True, False, False
    return list_state


@app.callback(
    Output({'type': 'container', 'id': ALL, 'specifier': 'disabled_cooling'},
           'style'),
    Input({'type': ALL, 'id': ALL, 'specifier': 'checklist_activate_cooling'}, 'value'),
    State({'type': 'container', 'id': ALL, 'specifier': 'disabled_cooling'},
          'id'),
    State({'type': 'container', 'id': ALL, 'specifier': 'disabled_cooling'},
          'style')
)
def tab3_callback_disabled_cooling(input1, ids, styles):
    len_val = len(ids)

    new_styles = {'pointer-events': 'none', 'opacity': '0.4'}

    if input1[0] == [1]:
        list_state = [{}] * len_val
    else:
        list_state = [new_styles] * len_val
    return list_state


@app.callback(
    Output({'type': 'container', 'id': ALL, 'specifier': 'visibility'},
           'style'),
    Input({'type': 'input', 'id': ALL, 'specifier': 'dropdown_activate'},
          'value'),
    State({'type': 'input', 'id': ALL, 'specifier': 'dropdown_activate'},
          'options')
)
def tab5_callback_visibility(inputs, options):
    list_options = []
    for opt in options:
        list_options.extend([inside['value'] for inside in opt])
        # [[opt1, opt2],[opt3, opt4, opt5]] turns into
        # [opt1, opt2, opt3, opt4, opt5]

    for inp in inputs:
        #  Eliminate/replace chosen value with 'chose' for later
        list_options = \
            list(map(lambda item: item.replace(inp, 'chosen'),
                     list_options))

    for num, lst in enumerate(list_options):
        if lst == 'chosen':
            # style = None / CSS revert to initial; {display:initial}
            list_options[num] = None
        else:
            # CSS for hiding div
            list_options[num] = {'display': 'none'}

    return list_options


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)

import base64
import io
import os
from dash import dash_table
import numpy as np
import pandas as pd
import pickle
import copy
import sys
import json
import dash
from dash_extensions.enrich import Output, Input, State, ALL, dcc, \
    EnrichedOutput, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import data_conversion as dc, modal_functions as mf, \
    study_functions as sf, simulation_api as sim
from .layout import layout_functions as lf

import pemfc
from pemfc.src import interpolation as ip
from pemfc_gui import data_transfer
from tqdm import tqdm
from decimal import Decimal
from .dash_app import app

tqdm.pandas()


# Callback functions (all other  functions organized in important modules)
# -----------------------------------------------------------------------------
@app.callback(
    Output('pbar', 'value'),
    Output('pbar', 'label'),
    Output('pbar', 'color'),
    Input('timer_progress', 'n_intervals'),
    prevent_initial_call=True)
def progress_bar(*args) -> (float, str):
    """
    https://towardsdatascience.com/long-callbacks-in-dash-web-apps-72fd8de25937
    """

    percent = 0.0
    try:
        with open('progress.txt', 'r') as file:
            str_raw = file.read()
        last_line = list(filter(None, str_raw.split('\n')))[-1]
        percent = float(last_line.split('%')[0])
    except FileNotFoundError as E:
        pass
    finally:
        text = f'{percent:.0f}%'
        if int(percent) == 100:
            color = "success"
        else:
            color = "primary"
        return percent, text, color


@app.callback(
    Output({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
    Output({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
    Output('pemfc_settings_file', 'data'),
    Output('df_input_store', 'data'),
    Output('study_table', 'children'),
    Input('initial_dummy', 'children'),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id')]
)
def initialization(dummy, value_list: list, multivalue_list: list,
                   id_list: list, multivalue_id_list: list):
    """
    Initialization
    """
    # Read pemfc default settings.json file
    # --------------------------------------
    try:
        # Initially get default simulation settings structure from
        # settings.json file in pemfc core module
        pemfc_base_dir = os.path.dirname(pemfc.__file__)
        with open(os.path.join(pemfc_base_dir, 'settings', 'settings.json')) \
                as file:
            base_settings = json.load(file)
        # Avoid local outputs from simulation
        base_settings['output']['save_csv'] = False
        base_settings['output']['save_plot'] = False
    except Exception as E:
        print(repr(E))
    base_settings = dc.store_data(base_settings)

    try:
        # Initially get default simulation input values from local
        # settings.json file
        with open(os.path.join('settings', 'settings.json')) \
                as file:
            input_settings = json.load(file)
        # Avoid local outputs from simulation
        input_settings['output']['save_csv'] = False
        input_settings['output']['save_plot'] = False
    except Exception as E:
        print(repr(E))
    gui_label_value_dict, _ = dc.settings_to_dash_gui(input_settings)

    # Update initial data input with "input_settings"
    # --------------------------------------
    # ToDO: This is a quick fix.
    # Solution should be: At GUI initialization, default values should be
    # taken from settings.json, NOT GUI-describing dictionaries.

    new_value_list, new_multivalue_list = \
        dc.update_gui_lists(gui_label_value_dict,
                            value_list, multivalue_list,
                            id_list, multivalue_id_list)

    # Read initial data input
    # --------------------------------------
    # Read data from input fields and save input in dict/dataframe
    # (one row "nominal")

    df_input = dc.process_inputs(new_value_list, new_multivalue_list,
                                 id_list, multivalue_id_list,
                                 returntype="DataFrame")
    df_input_store = dc.store_data(df_input)

    # Initialize study data table
    # -------------------------------------
    # Info: css stylesheet needed to be updated to show dropdown, see
    # https://community.plotly.com/t/resolved-dropdown-options-in-datatable-not-showing/20366/4

    # Dummy input
    empty_study_table = pd.DataFrame(dict([
        ('Parameter', list(df_input.columns)),
        ('Example', [str(x) for x in list(df_input.loc["nominal"])]),
        ('Variation Type', len(df_input.columns) * [None]),
        ('Values', len(df_input.columns) * [None])

        # ('ValueType', ["float", None, None, None])
    ]))

    table = dash_table.DataTable(
        id='study_data_table',
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px'
        },
        data=empty_study_table.to_dict('records'),
        columns=[
            {'id': 'Parameter', 'name': 'Parameter', 'editable': False},
            {'id': 'Example', 'name': 'Example', 'editable': False},
            {'id': 'Variation Type', 'name': 'Variation Type',
             'presentation': 'dropdown'},
            {'id': 'Values', 'name': 'Values'},
        ],

        editable=True,
        dropdown={
            'Variation Type': {
                'options': [
                    {'label': i, 'value': i}
                    for i in ["Values", "Percent (+/-)"]
                ]},
        },
        filter_action="native",
        sort_action="native",
        page_action='none',
        export_format='xlsx',
        export_headers='display',
        style_table={'height': '300px', 'overflowY': 'auto'}
    )
    return new_value_list, new_multivalue_list, base_settings, \
        df_input_store, table


@app.callback(
    [Output({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     Output({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     Output('upload-file', 'contents'),
     Output('modal_store', 'data')],
    Input('upload-file', 'contents'),
    [State('upload-file', 'filename'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id'),
     State('modal', 'is_open')],
    prevent_initial_call=True
)
def load_settings(contents, filename, value, multival, ids, ids_multival,
                  modal_state):
    if contents is None:
        raise PreventUpdate
    else:
        if 'json' in filename:
            try:
                settings_dict = dc.parse_contents(contents)
                gui_label_value_dict, error_list = \
                    dc.settings_to_dash_gui(settings_dict)

                new_value_list, new_multivalue_list = \
                    dc.update_gui_lists(gui_label_value_dict,
                                        value, multival, ids, ids_multival)

                if not error_list:
                    # All JSON settings match Dash IDs
                    modal_title, modal_body = mf.modal_process('loaded')
                    # Save modal input in a dict
                    modal_input = {'modal_title': modal_title,
                                   'modal_body': modal_body,
                                   'modal_state': not modal_state}
                    return new_value_list, new_multivalue_list, None, modal_input
                else:
                    # Some JSON settings do not match Dash IDs; return values
                    # that matched with Dash IDs
                    modal_title, modal_body = \
                        mf.modal_process('id-not-loaded', error_list)
                    # Save modal input in a dict
                    modal_input = {'modal_title': modal_title,
                                   'modal_body': modal_body,
                                   'modal_state': not modal_state}
                    return new_value_list, new_multivalue_list, None, modal_input
            except Exception as E:
                # Error / JSON file cannot be processed; return old value
                modal_title, modal_body = \
                    mf.modal_process('error', error=repr(E))
                # Save modal input in a dict
                modal_input = {'modal_title': modal_title,
                               'modal_body': modal_body,
                               'modal_state': not modal_state}
                return value, multival, None, modal_input
        else:
            # Not JSON file; return old value
            modal_title, modal_body = mf.modal_process('wrong-file')
            # Save modal input in a dict
            modal_input = {'modal_title': modal_title,
                           'modal_body': modal_body,
                           'modal_state': not modal_state}
            return value, multival, None, modal_input


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

    dict_data = dc.process_inputs(val1, val2, ids, ids2)  # values first

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

        with open(os.path.join('settings', 'settings.json')) \
                as file:
            settings = json.load(file)
        settings, _ = data_transfer.gui_to_sim_transfer(input_data, settings)

        return dict(content=json.dumps(settings, indent=2),
                    filename='settings.json')


@app.callback(
    [Output('modal-title', 'children'),
     Output('modal-body', 'children'),
     Output('modal', 'is_open')],
    Input('modal_store', 'data'),
    prevent_initial_call=True)
def convert_modal_input(modal_data):
    if modal_data is None:
        raise PreventUpdate
    modal_title = modal_data['modal_title']
    modal_body = modal_data['modal_body']
    modal_state = modal_data['modal_state']
    return modal_title, modal_body, modal_state


@app.callback(
    EnrichedOutput('df_result_data_store', 'data'),
    Output('df_input_store', 'data'),
    Output("spinner_run_single", 'children'),
    Output('modal_store', 'data'),
    Input("run_button", "n_clicks"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id'),
     State("pemfc_settings_file", "data"),
     State('modal', 'is_open')],
    prevent_initial_call=True)
def run_single_cal(n_click, inputs, inputs2, ids, ids2, settings, modal_state):
    """
    Changelog:

    """
    try:
        # Read pemfc settings.json from store
        settings = dc.read_data(settings)

        # Read data from input fields and save input in dict/dataframe
        # (one row "nominal")
        df_input = dc.process_inputs(inputs, inputs2, ids, ids2,
                                     returntype="DataFrame")
        df_input_raw = df_input.copy()

        # Create complete setting dict, append it in additional column
        # "settings" to df_input
        df_input = dc.create_settings(df_input, settings)

        # Run simulation
        df_result, _, err_modal, err_msg = sim.run_simulation(df_input)

        # Save results
        df_result_store = dc.store_data(df_result)
        df_input_store = dc.store_data(df_input_raw)

        if err_modal is not None:
            modal_title, modal_body = mf.modal_process(err_modal, error=err_msg)
            modal_state = not modal_state
        else:
            modal_title = None
            modal_body = None

        # Save modal input in a dict
        modal_input = {'modal_title': modal_title,
                       'modal_body': modal_body,
                       'modal_state': modal_state}
        return df_result_store, df_input_store, "", modal_input

    except Exception as E:

        modal_state = not modal_state
        modal_title, modal_body = \
            mf.modal_process('other-error', error=repr(E))

        # Save modal input in a dict
        modal_input = {'modal_title': modal_title,
                       'modal_body': modal_body,
                       'modal_state': modal_state}
        return None, None, "", modal_input


@app.callback(
    EnrichedOutput('df_result_data_store', 'data'),
    Output('df_input_store', 'data'),
    Output('spinner_ui', 'children'),
    Output('modal_store', 'data'),
    Input("btn_init_ui", "n_clicks"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id'),
     State('pemfc_settings_file', 'data'),
     State('modal', 'is_open')],
    prevent_initial_call=True)
def run_initial_ui_calculation(btn, inputs, inputs2, ids, ids2,
                               settings, modal_state):
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
    # Number of refinement steps
    n_refinements = 10

    try:
        # Progress bar init
        std_err_backup = sys.stderr
        file_prog = open('progress.txt', 'w')
        sys.stderr = file_prog

        # Read pemfc settings.json from store
        settings = dc.read_data(settings)

        # Read data from input fields and save input in dict/dataframe
        # (one row "nominal")
        df_input = dc.process_inputs(inputs, inputs2, ids, ids2,
                                     returntype="DataFrame")
        df_input_backup = df_input.copy()

        # Ensure DataFrame with double bracket
        # https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
        df_input_single = df_input.loc[["nominal"], :]
        max_i = sf.find_max_current_density(df_input_single, df_input, settings)

        # Reset solver settings
        df_input = df_input_backup.copy()

        # Prepare & calculate initial points
        df_results = sf.uicalc_prepare_initcalc(
            input_df=df_input, i_limits=[1, max_i],
            settings=settings)
        df_results, success, _, _ = sim.run_simulation(df_results)

        # First refinement steps
        for _ in range(n_refinements):
            df_refine = sf.uicalc_prepare_refinement(
                data_df=df_results, input_df=df_input, settings=settings)
            df_refine, success, _, _ = sim.run_simulation(
                df_refine, return_unsuccessful=False)
            df_results = pd.concat([df_results, df_refine], ignore_index=True)

        # Save results
        results = dc.store_data(df_results)
        df_input_store = dc.store_data(df_input_backup)

        # Close process bar files
        file_prog.close()
        sys.stderr = std_err_backup

        # Save modal input in a dict
        modal_input = {'modal_title': None,
                       'modal_body': None,
                       'modal_state': None}
        return results, df_input_store, ".", modal_input
    except Exception as E:
        modal_state = not modal_state
        modal_title, modal_body = \
            mf.modal_process('ui-error', error=repr(E))

        # Save modal input in a dict
        modal_input = {'modal_title': modal_title,
                       'modal_body': modal_body,
                       'modal_state': modal_state}
        return None, None, "", modal_input


@app.callback(
    EnrichedOutput('df_result_data_store', 'data'),
    Output('spinner_uirefine', 'children'),
    Output('modal_store', 'data'),
    Input("btn_refine_ui", "n_clicks"),
    State('df_result_data_store', 'data'),
    State('df_input_store', 'data'),
    State("pemfc_settings_file", "data"),
    State('modal', 'is_open'),
    prevent_initial_call=True)
def run_refine_ui(inp, state, state2, settings, modal_state):
    # Number of refinement steps
    n_refinements = 5

    try:
        # Progress bar init
        std_err_backup = sys.stderr
        file_prog = open('progress.txt', 'w')
        sys.stderr = file_prog

        # Read pemfc settings.json from store
        settings = dc.read_data(settings)

        # State-Store access returns None, I don't know why (FKL), workaround:
        df_results = dc.read_data(ctx.states["df_result_data_store.data"])
        df_nominal = dc.read_data(ctx.states["df_input_store.data"])

        # Refinement loop
        for _ in range(n_refinements):
            df_refine = sf.uicalc_prepare_refinement(
                data_df=df_results, input_df=df_nominal, settings=settings)
            df_refine, success, _, _ = sim.run_simulation(
                df_refine, return_unsuccessful=False)
            df_results = pd.concat([df_results, df_refine], ignore_index=True)

        # Save results
        results = dc.store_data(df_results)

        # Close process bar files
        file_prog.close()
        sys.stderr = std_err_backup

        # Save modal input in a dict
        modal_input = {'modal_title': None,
                       'modal_body': None,
                       'modal_state': None}
        return results, "", modal_input
    except Exception as E:
        modal_state = not modal_state
        modal_title, modal_body = \
            mf.modal_process('ui-error', error=repr(E))

        # Save modal input in a dict
        modal_input = {'modal_title': modal_title,
                       'modal_body': modal_body,
                       'modal_state': modal_state}
        return None, "", modal_input


@app.callback(Output('study_data_table', 'data'),
              Output('study_data_table', 'columns'),
              Output('modal_store', 'data'),
              Input('datatable-upload', 'contents'),
              State('datatable-upload', 'filename'),
              State('modal', 'is_open'),
              prevent_initial_call=True)
def update_studytable(contents, filename, modal_state):
    try:
        if contents is None:
            return [{}], []
        df = dc.parse_contents(contents, filename)
        # Save modal input in a dict
        modal_input = {'modal_title': None,
                       'modal_body': None,
                       'modal_state': None}
        return df.to_dict('records'), \
            [{"name": i, "id": i} for i in df.columns], modal_input
    except Exception as E:
        modal_title, modal_body = mf.modal_process('error-study-file')

        # Save modal input in a dict
        modal_input = {'modal_title': modal_title,
                       'modal_body': modal_body,
                       'modal_state': not modal_state}
        return None, None, modal_input


@app.callback(
    EnrichedOutput('df_result_data_store', 'data'),
    Output('df_input_store', 'data'),
    Output('spinner_study', 'children'),
    Output('modal_store', 'data'),
    Input("btn_study", "n_clicks"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id')],
    State("pemfc_settings_file", "data"),
    State("study_data_table", "data"),
    State("check_calc_ui", "value"),
    State("check_study_type", "value"),
    State('modal', 'is_open'),
    prevent_initial_call=True)
def run_study(btn, inputs, inputs2, ids, ids2, settings, tabledata,
              check_calc_ui, check_study_type, modal_state):
    """
    #ToDO Documentation

    Arguments
    ----------
    settings
    tabledata
    check_calc_ui:    Checkbox, if complete
    check_study_type:
    """
    err_modal = None
    err_msg = None
    try:
        variation_mode = "dash_table"

        # Calculation of polarization curve for each dataset?
        if isinstance(check_calc_ui, list):
            if "calc_ui" in check_calc_ui:
                ui_calculation = True
            else:
                ui_calculation = False
        else:
            ui_calculation = False

        # Number of refinement steps for ui calculation
        n_refinements = 10

        mode = check_study_type

        # Progress bar init
        std_err_backup = sys.stderr
        file_prog = open('progress.txt', 'w')
        sys.stderr = file_prog

        # Read pemfc settings.json from store
        settings = dc.read_data(settings)

        # Read data from input fields and save input in dict (legacy)
        # / pd.DataDrame (one row with index "nominal")
        df_input = dc.process_inputs(
            inputs, inputs2, ids, ids2, returntype="DataFrame")
        df_input_backup = df_input.copy()

        # Create multiple parameter sets
        if variation_mode == "dash_table":
            data = sf.variation_parameter(
                df_input, keep_nominal=False, mode=mode, table_input=tabledata)
        else:
            data = sf.variation_parameter(
                df_input, keep_nominal=False, mode=mode, table_input=None)
        varpars = list(data["variation_parameter"].unique())

        if not ui_calculation:
            # Create complete setting dict & append it in additional column
            # "settings" to df_input
            data = dc.create_settings(
                data, settings, input_cols=df_input.columns)
            # Run Simulation
            results, success, err_modal, err_msg = sim.run_simulation(data)
            results = dc.store_data(results)

        else:  # ... calculate pol. curve for each parameter set
            result_data = pd.DataFrame(columns=data.columns)

            # grouped_data = data.groupby(varpars, sort=False)
            # for _, group in grouped_data:
            for i in range(0, len(data)):
                try:
                    # Ensure DataFrame with double bracket
                    # https://stackoverflow.com/questions/20383647/pandas-selecting-by-label-sometimes-return-series-sometimes-returns-dataframe
                    # df_input_single = df_input.loc[[:], :]
                    max_i = sf.find_max_current_density(
                        data.iloc[[i]], df_input, settings)

                    # # Reset solver settings
                    # df_input = df_input_backup.copy()

                    success = False

                    # Prepare & calculate initial points
                    df_results = sf.uicalc_prepare_initcalc(
                        input_df=data.iloc[[i]], i_limits=[1, max_i],
                        settings=settings, input_cols=df_input.columns)
                    df_results, success, _, _ = sim.run_simulation(df_results)

                    if not success:
                        continue

                    # First refinement steps
                    for _ in range(n_refinements):
                        df_refine = sf.uicalc_prepare_refinement(
                            input_df=df_input, data_df=df_results,
                            settings=settings)
                        df_refine, success, _, _ = sim.run_simulation(
                            df_refine, return_unsuccessful=False)
                        df_results = pd.concat(
                            [df_results, df_refine], ignore_index=True)

                    result_data = pd.concat([result_data, df_results],
                                            ignore_index=True)
                except Exception as E:
                    err_modal = "generic-study-error"
                    pass

            results = dc.store_data(result_data)

        df_input_store = dc.store_data(df_input_backup)

        file_prog.close()
        sys.stderr = std_err_backup
    except Exception as E:
        modal_state = not modal_state
        modal_title, modal_body = \
            mf.modal_process("generic-study-error", error=repr(E))
        # Save modal input in a dict
        modal_input = {'modal_title': modal_title,
                       'modal_body': modal_body,
                       'modal_state': modal_state}
        return None, None, "", modal_input

    if err_modal is not None:
        modal_state = not modal_state
    modal_title, modal_body = \
        mf.modal_process(err_modal, error=repr(err_msg))

    # Save modal input in a dict
    modal_input = {'modal_title': modal_title,
                   'modal_body': modal_body,
                   'modal_state': modal_state}
    return results, df_input_store, '', modal_input


@app.callback(
    Output("download-results", "data"),
    Input("btn_save_res", "n_clicks"),
    State('df_result_data_store', 'data'),
    prevent_initial_call=True)
def save_results(inp, state):
    # State-Store access returns None, I don't know why (FKL)
    data = ctx.states["df_result_data_store.data"]
    with open('results.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dcc.send_file("results.pickle")


@app.callback(
    EnrichedOutput('df_result_data_store', 'data'),
    Input("load_res", "contents"),
    prevent_initial_call=True)
def load_results(content):
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
def figure_ui(inp1, inp2, dfinp):
    """
    Prior to plot: identification of same parameter sets with different
    current density. Those points will be connected and have identical color
    """

    # Read results
    results = dc.read_data(ctx.inputs["df_result_data_store.data"])
    df_nominal = dc.read_data(ctx.states["df_input_store.data"])
    results = results.loc[results["successful_run"] == True, :]
    results = results.drop(columns=['local_data'])

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Check for identical parameters, varying current density and voltage only
    group_columns = list(df_nominal.columns)
    group_columns.remove('simulation-current_density')
    group_columns.remove('simulation-average_cell_voltage')

    # Filter columns from results with NA or None values
    na_columns = results.isna().any()
    na_columns = na_columns[na_columns == True]
    drop_labels = list(na_columns.index)
    group_columns = [item for item in group_columns if item not in drop_labels]

    # Groupby fails, as data contains lists, which are not hashable, therefore conversion to tuple
    # see https://stackoverflow.com/questions/52225301/error-unhashable-type-list-while-using-df-groupby-apply
    # see https://stackoverflow.com/questions/51052416/pandas-dataframe-groupby-into-list-with-list-in-cell-data
    # results_red = results.loc[:, df_nominal.columns].copy()
    results = results.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
    grouped = results.groupby(group_columns, sort=False)

    for _, group in grouped:
        group.sort_values(
            "simulation-current_density", ignore_index=True, inplace=True)
        group["Current Density"] = group["global_data"].apply(
            lambda x: x["Average Current Density"]["value"] if (x is not None) else None)
        group["Voltage"] = group["global_data"].apply(
            lambda x: x["Stack Voltage"]["value"] if (x is not None) else None)
        group["Power"] = group["global_data"].apply(
            lambda x: x["Stack Power"]["value"] if (x is not None) else None)

        # Add traces
        if "variation_parameter" in group.columns:
            # Variation parameter can be one parameter or multiple parameter
            # separated by ",".
            varpar = group["variation_parameter"][0]
            try:
                if varpar.find(',') == -1:  # no param separator -> one param
                    setname = f"{varpar}: {group[varpar][0]}"
                    # Add figure title
                    fig.update_layout(title_text=f"Current-Voltage Curve")
                else:  # parameter separator found, multiple parameter...
                    list_varpar = [par for par in varpar.split(',')]
                    if len(list_varpar) > 3:  # don't plot legend
                        fig.update_layout(showlegend=False,
                                          title_text="Current-Voltage Curve")
                        setname = ""
                        for vp in list_varpar:
                            if isinstance(group[vp][0], tuple):
                                setname += \
                                    f'{vp}:[{Decimal(group[vp][0][0]):.3E}, '
                                setname += \
                                    f'{Decimal(group[vp][0][1]):.3E}] , <br>'
                            else:
                                setname += \
                                    f'{vp}:{Decimal(group[vp][0]):.3E} , <br>'

                    else:
                        fig.update_layout(
                            title_text=f"Current-Voltage Curve, "
                                       f"Variation parameter: <br> "
                                       f"{[par for par in varpar.split(',')]}")
                        setname = ""
                        for n, vp in enumerate(list_varpar):
                            if isinstance(group[vp][0], tuple):
                                setname += \
                                    f'par{n}:[{Decimal(group[vp][0][0]):.3E}, '
                                setname += \
                                    f'{Decimal(group[vp][0][1]):.3E}] , <br>'
                            else:
                                setname += f'{Decimal(group[vp][0]):.3E} , <br>'

                    setname = setname[:-6]

            except Exception as E:
                setname = "tbd"
                # Add figure title
                fig.update_layout()

        else:
            # Add figure title
            fig.update_layout()
            setname = ""

        if len(group) > 1:
            fig.add_trace(
                go.Scatter(x=list(group["Current Density"]),
                           y=list(group["Voltage"]), name=f"{setname}",
                           mode='lines+markers'), secondary_y=False)
        else:
            fig.add_trace(
                go.Scatter(x=list(group["Current Density"]),
                           y=list(group["Voltage"]), name=f"{setname}",
                           mode='markers'), secondary_y=False)

    # # Set x-axis title
    # fig.update_xaxes(title_text="Current Density [A/m²]")
    #
    # # Set y-axes titles
    # fig.update_yaxes(title_text="Voltage [V]", secondary_y=False)
    # fig.update_yaxes(title_text="Power [W]", secondary_y=True)

    x_title = 'Current Density / A/m²'
    y_title = 'Voltage / V'
    layout = go.Layout(
        font={'color': 'black', 'family': 'Arial'},
        # title='Local Results in Heat Map',
        titlefont={'size': 11, 'color': 'black'},
        xaxis={'tickfont': {'size': 11}, 'titlefont': {'size': 14},
               'title': x_title},
        yaxis={'tickfont': {'size': 11}, 'titlefont': {'size': 14},
               'title': y_title},
        margin={'l': 100, 'r': 20, 't': 20, 'b': 20})
    fig.update_layout(layout, hoverlabel=dict(namelength=-1))
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
    results = ctx.inputs["df_result_data_store.data"]
    if results is None:
        raise PreventUpdate
    results = dc.read_data(results)
    result_set = results.iloc[0]
    global_result_dict = result_set["global_data"]
    if global_result_dict is None:
        raise PreventUpdate
    names = list(global_result_dict.keys())
    values = [f"{v['value']:.3e}" for k, v in global_result_dict.items()]
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
    results = ctx.inputs["df_result_data_store.data"]
    if results is None:
        raise PreventUpdate
    results = dc.read_data(results)
    result_set = results.iloc[0]
    local_data = result_set["local_data"]
    if local_data is None:
        raise PreventUpdate
    options = \
        [{'label': key, 'value': key} for key in local_data
         if 'xkey' in local_data[key]
         and local_data[key]['xkey'] == 'Channel Location']
    value = options[0]['value']
    return options, value


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
    results = ctx.inputs["df_result_data_store.data"]
    if results is None:
        raise PreventUpdate
    results = dc.read_data(results)
    result_set = results.iloc[0]
    local_data = result_set["local_data"]
    if local_data is None:
        raise PreventUpdate

    options = [{'label': key, 'value': key} for key in local_data]
    value = options[0]['value']
    return options, value


@app.callback(
    [Output('dropdown_heatmap_2', 'options'),
     Output('dropdown_heatmap_2', 'value'),
     Output('dropdown_heatmap_2', 'style')],
    [Input('dropdown_heatmap', 'value'),
     Input('df_result_data_store', 'data')]
)
def get_dropdown_options_heatmap_2(dropdown_key, results):
    # if dropdown_key is None or results is None:
    #     raise PreventUpdate
    # else:
    # Read results
    if dropdown_key is None:
        raise PreventUpdate
    else:
        results = ctx.inputs["df_result_data_store.data"]
        if results is None:
            raise PreventUpdate
        else:
            return lf.conditional_dropdown_menu(dropdown_key, results)


@app.callback(
    [Output('dropdown_line2', 'options'),
     Output('dropdown_line2', 'value'),
     Output('dropdown_line2', 'style')],
    [Input('dropdown_line', 'value'),
     Input('df_result_data_store', 'data')],
    prevent_initial_call=True
)
def get_dropdown_options_line_graph_2(dropdown_key, results):
    # if dropdown_key is None or results is None:
    #     raise PreventUpdate
    # else:
    # Read results
    if dropdown_key is None:
        raise PreventUpdate
    else:
        results = ctx.inputs["df_result_data_store.data"]
        if results is None:
            raise PreventUpdate
        else:
            return lf.conditional_dropdown_menu(dropdown_key, results)


@app.callback(
    Output("heatmap_graph", "figure"),
    [Input('dropdown_heatmap', 'value'),
     Input('dropdown_heatmap_2', 'value'),
     Input('df_result_data_store', 'data')],
    prevent_initial_call=True
)
def update_heatmap_graph(dropdown_key, dropdown_key_2, results):
    # if dropdown_key is None or results is None:
    #     raise PreventUpdate
    # else:

    # Read results
    results = ctx.inputs["df_result_data_store.data"]
    if results is None:
        raise PreventUpdate
    results = dc.read_data(results)
    result_set = results.iloc[0]
    local_data = result_set["local_data"]
    if local_data is None:
        raise PreventUpdate

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

    height = 800
    # width = 500

    font_props = lf.graph_font_props

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
     Input('line_graph', 'restyleData'),
     Input('df_result_data_store', 'data')],
    prevent_initial_call=True
)
def update_line_graph(drop1, drop2, checklist, select_all_clicks,
                      clear_all_clicks, restyle_data, results):
    ctx_triggered = dash.callback_context.triggered[0]['prop_id']

    # Read results
    results = ctx.inputs["df_result_data_store.data"]
    if results is None:
        raise PreventUpdate
    results = dc.read_data(results)
    result_set = results.iloc[0]
    local_data = result_set["local_data"]
    if local_data is None:
        raise PreventUpdate

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
    elif drop2 is not None and 'value' in local_data[drop1][drop2]:
        yvalues = np.asarray(local_data[drop1][drop2]['value'])
    else:
        raise PreventUpdate

    n_y = np.asarray(yvalues).shape[0]
    n_x = np.asarray(yvalues).shape[-1]
    if x_key in local_data:
        xvalues = np.asarray(local_data[x_key]['value'])
        if len(xvalues) == n_y + 1:
            xvalues = np.round(ip.interpolate_1d(xvalues), 8)
    else:
        xvalues = np.asarray(list(range(n_x)))

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

    if checklist is not None:
        if 'clear_all_button.n_clicks' in ctx_triggered:
            value = []
        elif 'data_checklist.value' in ctx_triggered:
            value = checklist
        elif 'line_graph.restyleData' in ctx_triggered:
            read = restyle_data[0]['visible']
            if len(read) == 1:
                cell_name = cells[restyle_data[1][0]]['name']
                if read[0] is True:
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
    return fig, cells, options, value


@app.callback(
    Output("savefile-data", "data"),
    Input('export_csv', 'n_clicks'),
    State('line_graph', 'figure'),
    prevent_initial_call=True
)
def export_data(n_clicks, fig):

    # Get all data in figure
    data = fig['data']
    # Filter only visible data
    visible_data = [item for item in data if item['visible'] is True]
    # Format as Pandas DataFrame
    first_header_row = [fig['layout']['xaxis']['title']['text']] \
        + [fig['layout']['yaxis']['title']['text']
           for i in range(len(visible_data))]
    second_header_row = [''] + [item['name'] for item in visible_data]

    pure_data = np.asarray(
        [visible_data[0]['x']]
        + [item['y'] for item in visible_data]).transpose()
    df = pd.DataFrame(
        pure_data,
        columns=pd.MultiIndex.from_arrays(
            [first_header_row, second_header_row]))

    return dcc.send_data_frame(df.to_csv, "results.csv")


@app.callback(
    Output({'type': ALL, 'id': ALL, 'specifier': 'disable_basewidth'},
           'disabled'),
    Input({'type': ALL, 'id': ALL, 'specifier': 'dropdown_activate_basewidth'},
          'value'))
def disabled_callback(value):
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
def activate_column(input1, input2):
    len_state = len(input2)
    list_state = [True for x in range(len_state)]  # disable=True for all inputs
    for num, val in enumerate(input1):  # 3 inputs in input1 for 3 rows
        if val == [1]:
            list_state[0 + num] = list_state[3 + num] = list_state[15 + num] = \
                list_state[18 + num] = list_state[30 + num] = False
            if input2[3 + num] == 'circular':
                list_state[6 + num], list_state[9 + num], \
                    list_state[12 + num] = False, True, True
            else:
                list_state[6 + num], list_state[9 + num], \
                    list_state[12 + num] = True, False, False
            if input2[18 + num] == 'circular':
                list_state[21 + num], list_state[24 + num], \
                    list_state[27 + num] = False, True, True
            else:
                list_state[21 + num], list_state[24 + num], \
                    list_state[27 + num] = True, False, False
    return list_state


@app.callback(
    Output({'type': 'container', 'id': ALL, 'specifier': 'disabled_cooling'},
           'style'),
    Input({'type': ALL, 'id': ALL, 'specifier': 'checklist_activate_cooling'},
          'value'),
    State({'type': 'container', 'id': ALL, 'specifier': 'disabled_cooling'},
          'id'),
    State({'type': 'container', 'id': ALL, 'specifier': 'disabled_cooling'},
          'style')
)
def disabled_cooling(input1, ids, styles):
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
def visibility(inputs, options):
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

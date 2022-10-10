# import pathlib
import re
import copy
import numpy as np
import json
import os

import dash
from dash_extensions.enrich import Output, Input, State, ALL, html, dcc, \
    ServersideOutput
from dash import dash_table as dt
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import pemfc
from pemfc.src import interpolation as ip
from pemfc.data import input_dicts
from pemfc import main_app
from pemfc_gui import data_transfer
import pemfc_gui.input as gui_input

from . import dash_functions as df, dash_layout as dl, \
    dash_modal as dm
from .dash_tabs import tab3
from .dash_tabs import tab1, tab2, tab4, tab6, tab5
from pemfc_dash.dash_app import app

tab_list = [tab1.tab_layout, tab2.tab_layout, tab3.tab_layout,
            tab4.tab_layout, tab5.tab_layout, tab6.tab_layout]

server = app.server

app._favicon = 'logo-zbt.ico'
app.title = 'PEMFC Model'

app.layout = dbc.Container(
    [html.Div(  # HEADER
        [
         html.Div(
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
         html.Div(
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
             # width=8, align='center',
             style={'justify-content': 'space-evenly'},
             className='col-12 col-lg-8 mb-2'),
        ],
        id="header",
        className='row'
        ),
     dcc.Store(id="input_data"),
     dbc.Spinner(dcc.Store(id='result_data_store'), fullscreen=True,
                 spinner_class_name='loading_spinner',
                 fullscreen_class_name='loading_spinner_bg'),
     dcc.Store(id='signal'),

     # empty Div to trigger javascript file for graph resizing
     html.Div(id="output-clientside"),
     # modal for any warning
     dm.create_modal(),
     html.Div(  # MIDDLE
         [html.Div(  # LEFT MIDDLE
             [html.Div(  # LEFT MIDDLE MIDDLE
                  [dl.tab_container(
                          tab_list, label=
                          [k['title'] for k in gui_input.main_frame_dicts],
                          ids=['tab{}'.format(num + 1) for num in
                               range(len(gui_input.main_frame_dicts))])],
              id='setting_container',  # style={'flex': '1'}
              ),
              html.Div(   # LEFT MIDDLE BOTTOM
                  [html.Div(
                       [html.Div(
                           [
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
                            html.Button('Run Simulation', id='run_button',
                                        className='settings_button',
                                        style={'display': 'flex'})
                            ],
                           style={'display': 'flex',
                                  'flex-wrap': 'wrap',
                                  # 'flex-direction': 'column',
                                  # 'margin': '5px',
                                  'justify-content': 'space-evenly'}
                       ),
                        # dc.collapses
                       ],
                       className='neat-spacing')], style={'flex': '1'},
                  id='load_save_setting', className='pretty_container')],
             id="left-column", className='col-12 col-lg-4 mb-2'),

          html.Div(  # RIGHT MIDDLE
              [html.Div(
                        [html.Div('Global Results', className='title'),
                            dt.DataTable(id='global_data_table',
                                         editable=True,
                                         column_selectable='multi')],
                        id='div_global_table',
                        className='pretty_container',
                        style={'overflow': 'auto'}),
               html.Div(
                   [html.Div('Heatmap', className='title'),
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

               html.Div(
                 [html.Div('Plots', className='title'),
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
                  html.Div(
                      [html.Div(
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
                                    label="Select Cells"),),
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
     # html.Div(
     #     [],
     #     style={'position': 'relative',
     #            'margin': '0 0.05% 0 0.7%'})
     html.Div(dt.DataTable(id='table', editable=True,
                           column_selectable='multi'),
             # columns=[{'filter_options': 'sensitive'}]),
              id='div_table', style={'overflow': 'auto',
                                     'position': 'relative',
                                     # 'margin': '0 0.05% 0 0.7%'
                                     },
              className='pretty_container'),
     html.Div(
         html.Div(
            [

             html.A('Source code:'),
             html.A('web interface',
                    href='https://www.github.com/zbt-tools/pemfc-dash-app',
                    target="_blank"),
             html.A("fuel cell model",
                    href='https://www.github.com/zbt-tools/pemfc-core',
                    target="_blank"),
            ],
            id='github_links',
            style={'overflow': 'auto',
                   'position': 'relative',
                   'justify-content': 'space-evenly',
                   'align-items': 'center',
                   'min-width': '30%',
                   'display': 'flex'},
         ),
          # columns=[{'filter_options': 'sensitive'}]),
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


@app.callback(
    ServersideOutput("result_data_store", "data"),
    Output('modal-title', 'children'),
    Output('modal-body', 'children'),
    Output('modal', 'is_open'),
    Input("signal", "data"),
    State('input_data', 'data'),
    # interval=1e10,
    State('modal', 'is_open'),
    # running=[(Output("run_button", "disabled"), True, False)],
    prevent_initial_call=True
)
def run_simulation(signal, input_data, modal_state):
    if signal is None:
        raise PreventUpdate
    try:
        pemfc_base_dir = os.path.dirname(pemfc.__file__)
        with open(os.path.join(pemfc_base_dir, 'settings', 'settings.json')) \
                as file:
            settings = json.load(file)
        settings, name_lists = \
            data_transfer.gui_to_sim_transfer(input_data, settings)
        global_data, local_data, sim = main_app.main(settings=settings)
    except Exception as E:
        modal_title, modal_body = \
            dm.modal_process('input-error', error=repr(E))
        return None, modal_title, modal_body, not modal_state
    return [global_data[0], local_data[0]], None, None, modal_state

# def try_simulation_store(**kwargs):
#     try:
#         results = simulation_store(**kwargs)
#     except Exception as E:
#         raise PreventUpdate
#     return results


@app.callback(
    [Output('input_data', 'data'),
     Output('signal', 'data')],
    Input("run_button", "n_clicks"),
    [State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'value'),
     State({'type': 'input', 'id': ALL, 'specifier': ALL}, 'id'),
     State({'type': 'multiinput', 'id': ALL, 'specifier': ALL}, 'id')]
)
def generate_inputs(n_click, inputs, inputs2, ids, ids2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'run_button' in changed_id and n_click is not None:
        dict_data = df.process_inputs(inputs, inputs2, ids, ids2)

        input_data = {}
        for k, v in dict_data.items():
            input_data[k] = {'sim_name': k.split('-'), 'value': v}
        return input_data, n_click
    else:
        raise PreventUpdate


@app.callback(
    [Output('global_data_table', 'columns'),
     Output('global_data_table', 'data'),
     Output('global_data_table', 'export_format')],
    Input('result_data_store', 'data'),
    prevent_initial_call=True
)
def global_outputs_table(results):
    if results is None:
        raise PreventUpdate
    else:
        global_result_dict = results[0]
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
    Input('result_data_store', 'data'),
    prevent_initial_call=True
)
def get_dropdown_options_heatmap(results):
    if results is None:
        raise PreventUpdate
    else:
        local_data = results[1]
        values = [{'label': key, 'value': key} for key in local_data
                  if 'xkey' in local_data[key]
                  and local_data[key]['xkey'] == 'Channel Location']
        return values, 'Current Density'


@app.callback(
    [Output('dropdown_line', 'options'),
     Output('dropdown_line', 'value')],
    Input('result_data_store', 'data'),
    prevent_initial_call=True
)
def get_dropdown_options_line_graph(results):
    if results is None:
        raise PreventUpdate
    else:
        local_data = results[1]
        values = [{'label': key, 'value': key} for key in local_data]
        return values, 'Current Density'


@app.callback(
    [Output('dropdown_heatmap_2', 'options'),
     Output('dropdown_heatmap_2', 'value'),
     Output('dropdown_heatmap_2', 'style')],
    [Input('dropdown_heatmap', 'value'),
     Input('result_data_store', 'data')]
)
def get_dropdown_options_heatmap_2(dropdown_key, results):
    if dropdown_key is None or results is None:
        raise PreventUpdate
    else:
        local_data = results[1]
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
    State('result_data_store', 'data'),
    prevent_initial_call=True
)
def get_dropdown_options_line_graph_2(dropdown_key, results):
    if dropdown_key is None or results is None:
        raise PreventUpdate
    else:
        local_data = results[1]
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
    State('result_data_store', 'data'),
    prevent_initial_call=True
)
def update_heatmap_graph(dropdown_key, dropdown_key_2, results):
    if dropdown_key is None or results is None:
        raise PreventUpdate
    else:
        local_data = results[1]

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
    State('result_data_store', 'data'),
    prevent_initial_call=True
)
def update_line_graph(drop1, drop2, checklist, select_all_clicks,
                      clear_all_clicks, restyle_data, results):
    ctx = dash.callback_context.triggered[0]['prop_id']
    if drop1 is None or results is None:
        raise PreventUpdate
    else:

        fig = go.Figure()
        local_data = results[1]

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
            if 'clear_all_button.n_clicks' in ctx:
                fig.for_each_trace(
                    lambda trace: trace.update(visible='legendonly'))
                return fig, cells, options, []
            elif 'data_checklist.value' in ctx:
                fig.for_each_trace(
                    lambda trace: trace.update(
                        visible=True) if trace.name in checklist
                    else trace.update(visible='legendonly'))
                return fig, cells, options, checklist
            elif 'line_graph.restyleData' in ctx:
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
     State('result_data_store', 'data'),
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
        local_data = results[1]
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
            return index+columns, data, 'csv', appended
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
                      'id': 'Cell {}'.format(d) + '-'+str(appended)}
                     for d in digit_list]
                new_columns = table_columns + app_columns
                app_datas = \
                    [{**{x_key: cell},
                      **{cells_data[k]['name']+'-'+str(appended):
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
                                dict_ids2[k+f'_{num}'] = df.check_ifbool(val)
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
     State({'type': 'multiinput', 'id': ALL, 'specifier':  ALL}, 'id')],
    prevent_initial_call=True,
)
def save_settings(n_clicks, val1, val2, ids, ids2):
    ctx = dash.callback_context.triggered[0]['prop_id']
    if 'save-button.n_clicks' in ctx:
        dict_data = df.process_inputs(val1, val2, ids, ids2)  # values first
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


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)


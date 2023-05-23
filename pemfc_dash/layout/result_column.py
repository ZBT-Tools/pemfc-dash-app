from dash_extensions.enrich import html, dcc
import dash_bootstrap_components as dbc
from dash import dash_table as dt

# Result column: graphs, plots (RIGHT CENTER)
result_column = html.Div([
    html.Div(
        [html.Div('Current-Voltage Curve', className='title'),
         dcc.Graph(id='ui')],
        id='div_ui',
        className='pretty_container',
        style={'overflow': 'auto'}),
    html.Div(
        [html.Div('Global Results (for Study only first dataset shown)',
                  className='title'),
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
                    id='div_results_dropdown_2', )],
            style={'display': 'flex',
                   'flex-direction': 'row',
                   'flex-wrap': 'wrap',
                   'justify-content': 'left'}),
        # RIGHT MIDDLE BOTTOM
        dbc.Spinner(dcc.Graph(id="heatmap_graph"),
                    spinner_class_name='loading_spinner',
                    fullscreen_class_name='loading_spinner_bg')],
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
                [html.Div(
                     [html.Div(
                         children=dbc.DropdownMenu(
                             id='checklist_dropdown',
                             children=[
                                 dbc.Checklist(
                                     id='data_checklist',
                                     # input_checked_class_name='checkbox',
                                     style={'max-height': '400px',
                                            'overflow': 'auto'})],
                             toggle_style={
                                 'textTransform': 'none',
                                 'background': '#fff',
                                 'border': '#ccc',
                                 'letter-spacing': '0',
                                 'font-size': '11px'},
                             align_end=True,
                             toggle_class_name='dropdown_input',
                             label="Select Cells"), ),
                         html.Button('Clear All', id='clear_all_button',
                                     className='local_data_buttons'),
                         html.Button('Select All',
                                     id='select_all_button',
                                     className='local_data_buttons'),
                         html.Button('Export',
                                     id='export_csv',
                                     className='local_data_buttons'),
                         dcc.Download(id="savefile-data")],
                     style={'display': 'flex',
                            'flex-wrap': 'wrap',
                            'margin-bottom': '5px'})],
            # style={'width': '200px'}
            )],
            style={'display': 'flex', 'flex-direction': 'column',
                   'justify-content': 'left'}),
        dbc.Spinner(dcc.Graph(id='line_graph'),
                    spinner_class_name='loading_spinner',
                    fullscreen_class_name='loading_spinner_bg')],
        className="pretty_container",
        style={'display': 'flex',
               'flex-direction': 'column',
               'justify-content': 'space-evenly'}
    )],
    id='right-column', className='col-12 col-lg-8 mb-2')

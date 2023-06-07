from dash_extensions.enrich import html, dcc
import dash_bootstrap_components as dbc
from pemfc_dash.layout import layout_functions as lf


import pemfc_gui.input as gui_input


# Process bar components
pbar = dbc.Progress(id='pbar')
timer_progress = dcc.Interval(id='timer_progress',
                              interval=15000)


menu_column = html.Div([  # LEFT MIDDLE / (Menu Column)
    # Menu Tabs
    html.Div([
        lf.tab_container(gui_input.main_frame_dicts)],
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
                html.Button('Calc. Current-Voltage Curve',
                            id='btn_init_ui',
                            className='settings_button',
                            style={'display': 'flex'}),
                html.Button('Refine Curve',
                            id='btn_refine_ui',
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
        dcc.Markdown(
            '''
            ###### Parameter Study

            **Instruction**  The table below shows all parameter. For 
            each parameter either percentual deviation
            or multiple values can be given. Separate multiple values by 
            comma. Column "Example" shows example input and is not used 
            for calculation. 
            Only numeric parameter implemented yet.

            The table can be exported, modified in Excel & uploaded. 
            Reload GUI to restore table functionality after upload. 

            Below table, define study options.  
            '''),
        html.Div(id="study_table"),
        dcc.Upload(
            id='datatable-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '90%', 'height': '40px', 'lineHeight': '40px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center',
                'margin': '10px'
            },
        ),
        html.Div(
            dbc.Checklist(
                id="check_calc_ui",
                options=[{'label': 'Calc. Current-Voltage Curve',
                          'value': 'calc_ui'}])),
        html.Div(
            dbc.RadioItems(
                id="check_study_type",
                options=[{'label': 'Single Variation',
                          'value': 'single'},
                         {'label': 'Full Factorial',
                          'value': 'full'}],
                value='single',
                inline=True)),
        html.Div([
            html.Div([
                html.Button('Run Study', id='btn_study',
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
            html.Div(
                [html.Button('Plot', id='btn_plot',
                             className='settings_button',
                             style={'display': 'flex'}),
                 html.Button('Save Results', id='btn_save_res',
                             className='settings_button',
                             style={'display': 'flex'}),
                 dcc.Download(id="download-results"),

                 dcc.Upload(
                     id='load_res',
                     children=html.Button(
                         'Load Results', id='btn_load_res',
                         className='settings_button',
                         style={'display': 'flex'}))],
                style={'display': 'flex',
                       'flex-wrap': 'wrap',
                       'justify-content': 'space-evenly'}
            )],
            className='neat-spacing')], style={'flex': '1'},
        id='save_load_res', className='pretty_container'),
    html.Div([  # LEFT MIDDLE: Spinner
        html.Div(
            [html.Div(
                [dbc.Spinner(html.Div(id="spinner_run_single")),
                 dbc.Spinner(html.Div(id="spinner_ui")),
                 dbc.Spinner(html.Div(id="spinner_uirefine")),
                 dbc.Spinner(html.Div(id="spinner_study"))],

                # style={'display': 'flex',
                #       'flex-wrap': 'wrap',
                #       'justify-content': 'space-evenly'}
            )],
            className='neat-spacing')],
        style={'flex': '1'},
        id='spinner_bar',
        className='pretty_container'),
    # Progress Bar
    html.Div([
        # See: https://towardsdatascience.com/long-callbacks-in-dash-web-apps-72fd8de25937
        html.Div([
            html.Div([pbar, timer_progress])],
            className='neat-spacing', hidden='hidden')], style={'flex': '1'},
        id='progress_bar', className='pretty_container')],
    id="left-column", className='col-12 col-lg-4 mb-2')

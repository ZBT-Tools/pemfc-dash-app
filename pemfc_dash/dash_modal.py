"""
See: https://dash-bootstrap-components.opensource.faculty.ai/docs/components/modal/
"""

from dash import html
import dash_bootstrap_components as dbc


def create_modal(main_id='modal', title_id='modal-title', body_id='modal-body'):
    return html.Div(dbc.Modal(
        [dbc.ModalHeader(dbc.ModalTitle(id=title_id,
                                        style={'font-weight': 'bold',
                                               'font-size': '20px'})),
         dbc.ModalBody(id=body_id)],
        id=main_id, is_open=False, size="lg"))


def modal_process(error_type, error=''):
    if isinstance(error, list):
        ids_str = ', '.join([str(index) for index in error])
    else:
        ids_str = ''
    space = {'margin-top': '10px'}
    bold = {'font-weight': 'bold'}
    contents = \
        {
         'input-error':
            {'title': 'Simulation Error',
             'body':
                [html.Div("Simulation returned an error, "
                          "please check inputs!"),
                 html.Div(style=space),
                 html.Div(error, style=space)]},
         'loaded':
            {'title': 'JSON file has been loaded!',
             'body':
                  [html.Div("Data from JSON file has been loaded!"),
                   html.Div(style=space),
                   html.Div("You can click 'Run Simulation' to simulate the "
                            "loaded parameter or continue to change the loaded "
                            "parameter value", style=space)]},
         'id-not-loaded':
             {'title': 'Attention! Missing value for some component ID!',
              'body':
                  [html.Div("Data from JSON file has been loaded!"),
                   html.Div(style=space),
                   html.Div(
                      "However, Dash's component IDs below could not be found "
                      "or could not be matched with IDs from JSON file"),
                   html.Div(ids_str, style={**space, **bold}),
                   html.Div(style=space),
                   html.Div("You can still simulate the loaded parameter by "
                            "either clicking 'Run Simulation', ", style=space),
                   html.Div('change manually the unloaded parameter or '
                            'review the JSON file again!')]},
         'error':
             {'title': 'Error!',
              'body': [
                  html.Div('There has been an error while uploading the JSON '
                           'file!'),
                  html.Div(style=space),
                  html.Div('Please review the JSON file again or try '
                           'using another file!', style=space)],
              },
         'wrong-file':
             {'title': 'Error! Wrong File!',
              'body': [
                  html.Div('Upload function only accepts JSON file!'),
                  html.Div(style=space),
                  html.Div('Please select another file!', style=space)]}}

    return contents[error_type]['title'], contents[error_type]['body']

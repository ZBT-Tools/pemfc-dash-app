
from dash.dependencies import Input, Output, State, ALL  # ClientsideFunction
from dash import html

from pemfc_dash.dash_app import app
import pemfc_gui.input as gui_input
from pemfc_dash import dash_layout as dl


tab_layout = html.Div(dl.frame(gui_input.main_frame_dicts[4]))


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
        # [[opt1, opt2],[opt3, opt4, opt5]]

    for inp in inputs:
        list_options = \
            list(map(lambda item: item.replace(inp, 'replace'),
                     list_options))
        #  Eliminate/replace chosen value with 'replace' for later
    for num, lst in enumerate(list_options):
        if lst == 'replace':
            list_options[num] = None
        else:
            list_options[num] = {'display': 'none'}
    return list_options




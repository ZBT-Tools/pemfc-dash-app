
from dash.dependencies import Input, Output, ALL
from dash import html

from dash_app import app
import pemfc_gui.input as gui_input
import dash_layout as dl

tab_layout = html.Div(dl.frame(gui_input.main_frame_dicts[0]))


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

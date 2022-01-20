
from dash.dependencies import Input, Output, State, ALL  # ClientsideFunction

from dash import html

from dash_app import app
import pemfc_gui.input as gui_input
import dash_layout as dl

tab_layout = html.Div(dl.frame(gui_input.main_frame_dicts[2]))


@app.callback(
    Output({'type': ALL, 'id': ALL, 'specifier': 'disabled_cooling'},
           'disabled'),
    Input({'type': ALL, 'id': ALL, 'specifier':
           'checklist_activate_cooling'}, 'value'),
    State({'type': ALL, 'id': ALL, 'specifier': 'disabled_cooling'}, 'value')
)
def disabled_cooling(input1, value):
    len_val = len(value)
    if input1[0] == [1]:
        list_state = [False for x in range(len_val)]
    else:
        list_state = [True for x in range(len_val)]
    return list_state


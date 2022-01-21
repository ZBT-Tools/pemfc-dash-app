from dash.dependencies import Input, Output, State, ALL  # ClientsideFunction
from dash import html

from dash_app import app
import pemfc_gui.input as gui_input
import dash_layout as dl

tab_layout = html.Div(dl.frame(gui_input.main_frame_dicts[2]))


@app.callback(
    Output({'type': 'container', 'id': ALL, 'specifier': 'disabled_cooling'},
           'style'),
    Input({'type': ALL, 'id': ALL, 'specifier':
           'checklist_activate_cooling'}, 'value'),
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


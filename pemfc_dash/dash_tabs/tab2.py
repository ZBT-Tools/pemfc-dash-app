# global imports
from dash.dependencies import Input, Output, ALL  # ClientsideFunction
from dash import html

# local imports
from dash_app import app
import pemfc_gui.input as gui_input
import dash_layout as dl

tab_layout = html.Div(dl.frame(gui_input.main_frame_dicts[1]))


@app.callback(
    Output({'type': ALL, 'id': ALL, 'specifier': 'disabled_manifolds'},
           'disabled'),
    Input({'type': ALL, 'id': ALL,
           'specifier': 'checklist_activate_calculation'}, 'value'),
    Input({'type': ALL, 'id': ALL, 'specifier': 'disabled_manifolds'}, 'value'),

)
def activate_column(input1, input2):
    len_state = len(input2)
    list_state = [True for x in range(len_state)]
    for num, val in enumerate(input1):
        if val == [1]:
            list_state[0 + num] = list_state[3 + num] = list_state[15 + num] = \
                list_state[18 + num] = list_state[30 + num] = False
            if input2[3+num] == 'circular':
                list_state[6+num], list_state[9+num], list_state[12+num] = \
                 False, True, True
            else:
                list_state[6+num], list_state[9+num], list_state[12+num] = \
                    True, False, False
            if input2[18+num] == 'circular':
                list_state[21+num], list_state[24+num], list_state[27+num] = \
                 False, True, True
            else:
                list_state[21+num], list_state[24+num], list_state[27+num] = \
                    True, False, False
    return list_state


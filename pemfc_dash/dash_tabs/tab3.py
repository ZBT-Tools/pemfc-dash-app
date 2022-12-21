from dash_extensions.enrich import Output, Input, State, ALL, html
from ..dash_app import app


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


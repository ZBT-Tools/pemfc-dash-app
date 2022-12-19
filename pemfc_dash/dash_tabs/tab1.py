from dash_extensions.enrich import Output, Input, ALL
from ..dash_app import app

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

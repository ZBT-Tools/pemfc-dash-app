from dash import Dash, dash_table, html
import pandas as pd
from collections import OrderedDict
from dash.dependencies import Input, Output


app = Dash(__name__)



df = pd.DataFrame(OrderedDict([
    ('Parameter', [None, None, None, None]),
    ('Value', [None, None, None, None]),
]))

par = 100*["mem","par1"]


app.layout = html.Div([
    dash_table.DataTable(
        id='table-dropdown',
        data=df.to_dict('records'),
        columns=[
            {'id': 'Parameter', 'name': 'parameter', 'presentation': 'dropdown'},
            {'id': 'Value', 'name': 'values'},
        ],

        editable=True,
        dropdown={
            'Parameter': {
                'options': [
                    {'label': i, 'value': i}
                    for i in par
                ]
            },
        }
    ),
    html.Div(id='selection')
])



@app.callback(
    Output("selection",'children'),
    Input('table-dropdown', 'data'),
    prevent_initial_call=True

)
def printable(rows):
    print(rows)

if __name__ == '__main__':
    app.run_server(debug=True)

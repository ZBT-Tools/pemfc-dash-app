from dash_extensions.enrich import html
from pemfc_dash.dash_app import app

header = html.Div(
    [  # HEADER (Header row, logo and title)
        html.Div(  # Logo
            html.Div(
                html.Img(
                    src=app.get_asset_url("logo-zbt.png"),
                    id="zbt-image",
                    style={"object-fit": 'contain',
                           'position': 'center',
                           "width": "auto",
                           "margin": "auto"}),
                id="logo-container", className="pretty_container h-100",
                style={'display': 'flex', 'justify-content': 'center',
                       'align-items': 'center'}
            ),
            className='col-12 col-lg-4 mb-2'
        ),
        html.Div(  # Title
            html.Div(
                html.H3("Fuel Cell Stack Model",
                        style={"margin": "auto",
                               "min-height": "47px",
                               "font-weight": "bold",
                               "-webkit-text-shadow-width": "1px",
                               "-webkit-text-shadow-color": "#aabad6",
                               "color": "#0062af",
                               "font-size": "40px",
                               "width": "auto",
                               "text-align": "center",
                               "vertical-align": "middle"}),
                className="pretty_container h-100", id="title",
                style={'justify-content': 'center', 'align-items': 'center',
                       'display': 'flex'}),
            style={'justify-content': 'space-evenly'},
            className='col-12 col-lg-8 mb-2'),
    ],
    id="header",
    className='row')

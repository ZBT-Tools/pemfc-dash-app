import pathlib

from dash import html

from flask_caching import Cache

from pemfc_dash.dash_app import app
# from pemfc_dash.dash_tabs import tab1, tab2, tab3, tab4, tab5, tab6
#
#
# tabs_list = [tab1.tab_layout, tab2.tab_layout, tab3.tab_layout,
#              tab4.tab_layout, tab5.tab_layout, tab6.tab_layout]

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
# app = dash.Dash(__name__)

server = app.server


# Setup caching
CACHE_CONFIG = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)
# cache2 = Cache()
# cache2.init_app(app.server, config=CACHE_CONFIG)
# app._favicon = 'logo-zbt.ico'
# app.title = 'PEMFC Model'

app.layout = html.Div(
    [html.Div(  # HEADER
        [html.Div(
            html.Div(html.Img(
                    src=app.get_asset_url("logo-zbt.png"),
                    id="zbt-image",
                    style={  # "min-height": "60px",
                           "height": "auto",  # "60px",
                           "object-fit": 'contain',
                           'position': 'relative',
                           "width": "auto",
                           "margin": "auto"
                    })))])])

if __name__ == "__main__":
    # [print(num, x) for num, x in enumerate(dl.ID_LIST) ]
    # app.run_server(host='127.0.0.1', port=8080, debug=True, use_reloader=False)
    app.run_server(debug=True, use_reloader=False)

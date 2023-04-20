from dash_extensions.enrich import html


# Bottom row, links to GitHub,...
bottom = html.Div(
    html.Div(
        [html.A('Source code:'),
         html.A('web interface',
                href='https://www.github.com/zbt-tools/pemfc-dash-app',
                target="_blank"),
         html.A("fuel cell model",
                href='https://www.github.com/zbt-tools/pemfc-core',
                target="_blank")],
        id='github_links',
        style={'overflow': 'auto',
               'position': 'relative',
               'justify-content': 'space-evenly',
               'align-items': 'center',
               'min-width': '30%',
               'display': 'flex'}),
    id='link_container',
    style={'overflow': 'auto',
           'position': 'relative',
           'justify-content': 'center',
           'align-items': 'center',
           'display': 'flex'},
    className='pretty_container')

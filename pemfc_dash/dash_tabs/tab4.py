from dash_extensions.enrich import html

import pemfc_gui.input as gui_input
from .. import dash_layout as dl

tab_layout = html.Div(dl.frame(gui_input.main_frame_dicts[3]))


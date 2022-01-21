from dash import html

from dash_app import app
import pemfc_gui.input as gui_input
import dash_layout as dl

tab_layout = html.Div(dl.frame(gui_input.main_frame_dicts[3]))


from decimal import Decimal
import data_transfer
import base64
import io
import json
import pickle
import jsonpickle
from glom import glom
import pandas as pd

from .layout import layout_functions as dl


def store_data(data):
    """
    https://github.com/jsonpickle/jsonpickle, as json.dumps can only handle
    simple variables, no objects, DataFrames..
    Info: Eigentlich sollte jsonpickle reichen, um dict mit Klassenobjekten,
    in denen DataFrames sind, zu speichern, es gibt jedoch Fehlermeldungen.
    Daher wird Datenstruktur vorher in pickle (Binärformat)
    gespeichert und dieser anschließend in json konvertiert.
    (Konvertierung in json ist notwendig für lokalen dcc storage)
    """
    data = pickle.dumps(data)
    data = jsonpickle.dumps(data)

    return data


def read_data(data):
    # Read NH3 data from storage
    data = jsonpickle.loads(data)
    data = pickle.loads(data)

    return data


def create_settings(df_data: pd.DataFrame, settings,
                    input_cols=None) -> pd.DataFrame:
    # Create settings dictionary
    # If "input_cols" are given, only those will be used from "df_data".
    # Use case: df_data can contain additional columns as study information that needs
    # to be excluded from settings dict
    # -----------------------------------------------------------------------
    # Create object columns
    df_temp = pd.DataFrame(columns=["input_data", "settings"])
    df_temp['input_data'] = df_temp['input_data'].astype(object)
    df_temp['settings'] = df_temp['input_data'].astype(object)

    if input_cols is not None:
        df_data_red = df_data.loc[:, input_cols]
    else:
        df_data_red = df_data

    # Create input data dictionary (legacy)
    df_temp['input_data'] = df_data_red.apply(
        lambda row: {i: {'sim_name': i.split('-'), 'value': v}
                     for i, v in zip(row.index, row.values)}, axis=1)

    df_temp['settings'] = df_temp['input_data'].apply(
        lambda x: data_transfer.dict_transfer(x, settings)[0])
    data = df_data.join(df_temp)

    return data


def unstringify(val: str | list) -> (float | str | list):
    """
    Used to change any str value created by DBC.Input once initialised due to
    not defining the component as type Number.
    """
    if isinstance(val, str):
        if val.isdigit():
            return int(val)
        else:
            try:
                return float(val)
            except (ValueError, NameError):
                return val
    elif isinstance(val, list):
        try:
            return [float(v) for v in val]
        except ValueError:
            return val
    else:
        return val


def multi_inputs(dicts):
    """
    Deal with components that have multiple values and multiple IDs from
    id-value dicts
    (can be found inside def process_inputs; dicts pre-processed inside the
    function)
    """
    dict_list = {}
    for k, v in dicts.items():
        if k[-1:].isnumeric() is False:
            dict_list.update({k: v})
        else:
            if k[:-2] not in dict_list:
                dict_list.update({k[:-2]: v})
            elif k[:-2] in dict_list:
                if not isinstance(dict_list[k[:-2]], list):
                    new_list = [dict_list[k[:-2]]]
                else:
                    new_list = dict_list[k[:-2]]
                new_list.append(v)
                dict_list.update({k[:-2]: new_list})
    return dict_list


def parse_contents(contents, filename='json'):
    """
    #ToDo: rework

    Used in parsing contents from JSON file and process parsed data
    for each components
    (parsed data has to be in the order of initialised Dash IDs)
    #ToDo: check & rework

    contents:
    contents is a base64 encoded string that contains the files contents,
    no matter what type of file:
     text files, images, .zip files, Excel spreadsheets, etc.
     [https://dash.plotly.com/dash-core-components/upload]
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Assume that the user uploaded an Excel file
        return pd.read_excel(io.BytesIO(decoded))
    elif 'json' in filename:
        return json.load(io.StringIO(decoded.decode('utf-8')))
    else:
        raise NotImplementedError('No parsing for provided file type '
                                  'implemented')


def settings_to_dash_gui(settings: dict) -> (dict, list):
    """
    Convert settings from the hierarchical simulation input dictionary to
    a dictionary for the gui input combining all hierarchical keys to a single
    id key string and the entry value.
    Example:
    input: settings_dict = {'stack': {'cathode': {'channel': {'length': 0.5}}}}
    return: gui_dict = {'stack-cathode-channel-length': 0.5}
    """
    name_lists = [ids['id'].split('-') if ids['id'][-1:].isnumeric() is False
                  else ids['id'][:-2].split('-') for ids in dl.ID_LIST]
    error_list = []
    gui_dict = {}
    for n in name_lists:
        name_id = '-'.join(n)
        try:
            gui_dict.update({name_id: glom(settings, '.'.join(n))})
        except Exception as e:
            # print(e)
            error_list.append(name_id)
    return gui_dict, error_list


def update_gui_lists(id_value_dict: dict,
                     old_vals: list, old_multivals: list,
                     ids: list, ids_multival: list) -> (list, list):
    dict_ids = \
        {id_l: val for id_l, val
         in zip([id_l['id'] for id_l in ids], old_vals)}
    dict_ids_multival = \
        {id_l: val for id_l, val
         in zip([id_l['id'] for id_l in ids_multival], old_multivals)}

    id_match = set.union(set(dict_ids),
                         set([item[:-2] for item in dict_ids_multival]))

    for k, v in id_value_dict.items():
        if k in id_match:
            if isinstance(v, list):
                for num, val in enumerate(v):
                    dict_ids_multival[k + f'_{num}'] = check_if_bool(val)
            else:
                dict_ids[k] = check_if_bool(v)
        else:
            continue
    return list(dict_ids.values()), list(dict_ids_multival.values())


def check_if_bool(val):
    """
    Used for dcc.Checklist components when receiving value from its
    tkinter.CheckButton counterparts
    """
    if isinstance(val, bool):
        if val is True:
            return [1]
        else:
            return []
    else:
        return val


def process_inputs(inputs, multiinputs, id_inputs, id_multiinputs,
                   returntype="dict"):
    """

    Returns dict_data dictionary of format
        dict_data = {'stack-cell_number':1, ...}
    or pd.DataFrame with row "nominal", columns=['stack-cell_number',...]


    Used in matching key-value (id-value) in the order of the initialised
    Dash's IDs
    (multi inputs handle two value and has multiple IDs assigned to it)
    """
    new_inputs = []
    for val in inputs + multiinputs:
        new_val = unstringify(val)

        if isinstance(new_val, list):
            if len(new_val) == 0:
                new_val = bool(new_val)
            else:
                if len(new_val) == 1 and new_val[0] == 1:
                    new_val = bool(new_val)
        new_inputs.append(new_val)

    new_ids = [id_l['id'] for id_l in id_inputs] + \
              [id_l['id'] for id_l in id_multiinputs]

    dict_data = {}
    for id_l, v_l in zip(new_ids, new_inputs):
        dict_data.update({id_l: v_l})
    new_dict_data = multi_inputs(dict_data)

    if returntype == "dict":
        return new_dict_data
    elif returntype == "DataFrame":
        df_data = pd.DataFrame()
        for k, v in new_dict_data.items():
            # Info: pd.DataFrame.at instead of .loc, as .at can put lists into
            # df cell.
            # .loc can be used for passing values to more than one cell,
            # that's why passing lists is not possible.
            # Column must be of type object to accept list-objects
            # https://stackoverflow.com/questions/26483254/python-pandas-insert-list-into-a-cell
            df_data.at["nominal", k] = None
            df_data[k] = df_data[k].astype(object)
            df_data.at["nominal", k] = v
        return df_data


# def dict_inputs(value='', ids=''):
#     """
#     DEPRECATED
#     Create dictionary from given IDs and values (use def process_inputs)
#     """
#     data = {}
#     if value != '':
#         for id_l, val in zip(ids, value):
#             checked_val = unstringify(val)
#             data[id_l['id']] = checked_val[0]
#     else:
#         for val, id_l in enumerate(ids):
#             data[id_l['id']] = val
#     return data
#
#
# def list_dict_inputs(value='', ids=''):
#     """
#     DEPRECATED
#     Create list with nested dictionary from given IDs, and values
#     (use def process_inputs)
#     """
#     data = []
#     if value != '':
#         for id_l, val in zip(ids, value):
#             checked_val = unstringify(val)
#             data.append({id_l['id']: checked_val[0]})
#     else:
#         for val, id_l in enumerate(ids):
#             data.append({id_l['id']: val})
#     return data
#
#
# def multival_input(value, input_ids):
#     """
#     DEPRECATED
#     Process IDs-value into nested dictionary with 'sim_name' and 'value' as key
#     for further simulation (restructured under def compute_simulation)
#     """
#     inputs = [list(unstringify(v))[0] for v in value]
#     data_dict = {ids['id']: inp for ids, inp in zip(input_ids, inputs)}
#     set_list = list(set([k[:-2] for k, v in data_dict.items()]))
#     data = {k: [data_dict[k + '-z'], data_dict[k + '-x']] for k in set_list}
#
#     return {k: {'sim_name': k.split('-'), 'value': v} for k, v in data.items()}
#
#
# def dash_kwarg(inputs):
#     """
#     DEPRECATED
#     Used as a decorator to decorate callback functions (def compile_data) to
#     retrieve multiple values from multiple inputs in creating id-value
#     dictionary from given parameter
#     """
#
#     def accept_func(func):
#         @wraps(func)
#         def wrapper(*args):
#             input_names = [item.component_id for item in inputs]
#             kwargs_dict = dict(zip(input_names, args))
#             return func(**kwargs_dict)
#
#         return wrapper
#
#     return accept_func
#
#
# def compile_data(**kwargs):
#     """
#     DEPRECATED
#     Used in compiling data from each component into a dcc.Store for each Tab
#     """
#     dash_dict = collections.defaultdict(dict)
#     for k, v in kwargs.items():
#         t = dash_dict[k]['sim_name'] = k.split("-")
#         c_v = list(unstringify(v))
#         if t[-1] in ['cool_flow', 'cool_ch_bc', 'calc_distribution']:
#             dash_dict[k]['value'] = bool(c_v[0])
#         else:
#             dash_dict[k]['value'] = c_v[0]
#     return dict(dash_dict)


def generate_set_name(row):
    """
    Generates set names.
    Case distinction is required as value can be numeric or list (for value-pairs as
    conductivities)
    """
    var_pars = row['variation_parameter']
    setname = ""
    for vp in var_pars:
        if isinstance(row[vp], list):
            setname += f"{vp}: " + str(row[vp]) + ", "
        else:
            setname += f"{vp}: " + str(float(row[vp])) + ", "

    return setname[:-2]


def generate_ui_set_name(row):
    """
    Generates ui_curve set names.
    Case distinction is required as value can be numeric or list (for value-pairs as
    conductivities)
    """
    var_pars = row['variation_parameter']
    setname = ""
    for vp in var_pars:
        if isinstance(row[vp], list):
            setname += f"{vp}: " + str(row[vp]) + ", "
        else:
            setname += f"{vp}: " + str(float(row[vp])) + ", "

    setname = setname[:-2] + ", Current Density: " + str(round(float(
        row['simulation-current_density']),
        1)) + " A/m²"

    return setname


def float_to_str_format(val):
    """
    Formats float values
    """
    if isinstance(val, float):
        return f"{Decimal(val):.3E}"
    else:
        return val

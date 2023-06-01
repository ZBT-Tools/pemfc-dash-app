import pandas as pd
import numpy as np
import ast
from itertools import product

import data_transfer

from . import data_conversion as dc
from . import simulation_api as sim
# from main import create_settings


def uicalc_prepare_initcalc(input_df: pd.DataFrame, i_limits: list, settings,
                            input_cols=None) -> pd.DataFrame:
    """
    Initial calculations for polarization-curve
    input_df has to have only one row!
    @return:
    """

    data_df = pd.DataFrame(columns=input_df.columns)

    # Create initial input sets
    i_calc = np.linspace(i_limits[0], i_limits[1], 3)
    for i in i_calc:
        data_df.loc[i, :] = input_df.iloc[0, :]  # .loc["nominal", :]
        data_df.loc[i, "simulation-current_density"] = float(i)

    data_df = dc.create_settings(data_df, settings, input_cols=input_cols)

    data_df.loc[:, "u_pred"] = None
    data_df.loc[:, "u_pred_diff"] = None

    return data_df


def uicalc_prepare_refinement(
        data_df: pd.DataFrame, input_df: pd.DataFrame, settings):
    """
    ToDo: Documentation
    ToDO: Error handling: Handle None in results.
    Returns new calculation row(s)
    @return:
    """

    n = data_df.shape[0]
    new_data_df = pd.DataFrame(columns=data_df.columns)
    data_df.sort_values(
        "simulation-current_density", ignore_index=True, inplace=True)

    # First refinement,
    # set prediction = calculation & prediction different to zero
    if n == 3:
        data_df["u_pred"] = \
            data_df["global_data"].apply(lambda x: x["Stack Voltage"]["value"])
        data_df["u_pred_diff"] = 0
        refine = data_df

    else:  # refinement no. 2 onwards....

        # Calculate difference between calculations and prior predictions
        data_df["u_pred_diff"] = \
            abs(data_df["global_data"].apply(
                lambda x: x["Stack Voltage"]["value"]) - data_df["u_pred"])
        data_df["u_pred_diff"]= pd.to_numeric(data_df["u_pred_diff"])
        # Location of largest deviation
        idxmax = data_df["u_pred_diff"].idxmax()
        # "reset" predictions, as refinement will be performed now
        data_df.loc[idxmax, "u_pred"] = \
            data_df.loc[idxmax, "global_data"]["Stack Voltage"]["value"]

        refine = data_df.iloc[idxmax - 1: idxmax + 2]

    # DataFrame 'refine' has 3 rows. Two additional will be added inbetween:

    i_calc = refine["simulation-current_density"].to_list()
    u_calc = refine["global_data"].apply(
        lambda x: x["Stack Voltage"]["value"]).to_list()

    for idx in range(len(i_calc) - 1):
        i_new = (i_calc[idx] + i_calc[idx + 1]) / 2
        u_pred = (u_calc[idx] + u_calc[idx + 1]) / 2

        # duplicate random(here first) row and adjust current density)
        new_data_df.loc[i_new, :] = data_df.iloc[0, :]
        new_data_df.loc[i_new, "simulation-current_density"] = i_new
        new_data_df.loc[i_new, "u_pred"] = u_pred

    # Create settings out of (only) input columns
    new_data_df_red = new_data_df.loc[:, input_df.columns]
    # For legacy: Create "input_data"-dict,
    # as required for data_transfer.dict_transfer()
    new_data_df['input_data'] = new_data_df_red.apply(
        lambda row: {i: {'sim_name': i.split('-'), 'value': v}
                     for i, v in zip(row.index, row.values)}, axis=1)

    new_data_df['settings'] = new_data_df['input_data'].apply(
        lambda x: data_transfer.dict_transfer(x, settings)[0])

    return new_data_df


def variation_parameter(df_input: pd.DataFrame, keep_nominal=False,
                        mode="single", table_input=None) -> pd.DataFrame:
    """
    Function to create parameter sets.
    - variation of single parameter - ok
    - (single) variation of multiple parameters - ok
    - combined variation of multiple parameters
        - full factorial - ok

    Important: Change casting_func to int(),float(),... accordingly!
    """

    # Define parameter sets
    # -----------------------
    if not table_input:
        var_parameter = {
            "membrane-thickness":
                {"values": [0.25e-05, 4e-05], "casting": float},
            "cathode-electrochemistry-thickness_gdl":
                {"values": [0.00005, 0.0008], "casting": float},
            "anode-electrochemistry-thickness_gdl":
                {"values": [0.00005, 0.0008], "casting": float},
        }
    else:
        var_par_names = \
            [le["Parameter"] for le in table_input
             if le["Variation Type"] is not None]

        # Comment on ast...: ast converts string savely into int,float, list,...
        # It is required as input in DataTable is unspecified and need to be
        # cast appropriately.
        # On the other-hand after uploading table, values can be numeric,
        # which can cause Error.
        # Solution: Cast input always to string (required for uploaded data) and
        # then eval with ast
        var_par_values = \
            [ast.literal_eval(str(le["Values"])) for le in table_input
             if le["Variation Type"] is not None]
        var_par_variationtype = \
            [le["Variation Type"] for le in table_input
             if le["Variation Type"] is not None]
        var_par_cast = []

        # var_par_cast = [ast.literal_eval(str(le["Variation Type"]))
        # for le in table_input if le["Variation Type"] is not None]

        for le in var_par_values:
            le_type = type(le)
            if le_type == tuple:
                le_type = type(le[0])
            var_par_cast.append(le_type)

        # Caluclation of values for percent definitions
        processed_var_par_values = []
        for name, vls, vartype \
                in zip(var_par_names,
                       var_par_values, var_par_variationtype):
            nom = df_input.loc["nominal", name]
            if vartype == "Percent (+/-)":
                perc = vls
                if isinstance(nom, list):
                    # nomval = [cst(v) for v in nom]
                    processed_var_par_values.append(
                        [[v * (1 - perc / 100) for v in nom],
                         [v * (1 + perc / 100) for v in nom]])
                else:
                    # nomval = cst(nom)
                    processed_var_par_values.append([nom * (1 - perc / 100),
                                                     nom * (1 + perc / 100)])

            else:
                processed_var_par_values.append(list(vls))

        var_parameter = \
            {name: {"values": val} for name, val in
             zip(var_par_names, processed_var_par_values)}

    # Add informational column "variation_parameter"
    clms = list(df_input.columns)
    clms.extend(["variation_parameter"])
    data = pd.DataFrame(columns=clms)

    if mode == "single":  #
        # ... vary one variation_parameter, all other parameter nominal
        # (from GUI)
        for parname, attr in var_parameter.items():
            for val in attr["values"]:
                inp = df_input.copy()
                inp.loc["nominal", parname] = val
                inp.loc["nominal", "variation_parameter"] = parname
                data = pd.concat([data, inp], ignore_index=True)

    elif mode == "full":
        # see https://docs.python.org/3/library/itertools.html

        parameter_names = [key for key, val in var_parameter.items()]
        parameter_names_string = ",".join(parameter_names)
        parameter_values = [val["values"] for key, val in var_parameter.items()]
        parameter_combinations = list(product(*parameter_values))

        for combination in parameter_combinations:
            inp = df_input.copy()
            inp.loc["nominal", "variation_parameter"] = parameter_names_string
            for par, val in zip(parameter_names,
                                combination):
                inp.at["nominal", par] = val
            data = pd.concat([data, inp], ignore_index=True)

    if keep_nominal:
        data = pd.concat([data, df_input])

    return data


def find_max_current_density(data: pd.DataFrame, df_input, settings):
    """
    Note: Expects one row DataFrame

    @param data:
    @param df_input:
    @param settings:
    @return:
    """
    # Find highest current density
    # ----------------------------
    success = False
    u_min = 0.05  # V

    data_backup = data.copy()

    while not success:
        data = data_backup.copy()
        u_min += 0.05

        # Change solver settings temporarily to voltage control
        data.loc[:, "simulation-operation_control"] = "Voltage"
        data.loc[:, "simulation-average_cell_voltage"] = u_min

        # Create complete setting dict, append it in additional column
        # settings" to df_input
        data = dc.create_settings(data, settings, input_cols=df_input.columns)

        # Run simulation
        df_result, success, _, _ = sim.run_simulation(data)

    max_i = df_result["global_data"].iloc[0]["Average Current Density"]["value"]

    return max_i

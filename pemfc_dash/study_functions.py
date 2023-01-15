import pandas as pd
import numpy as np
from pemfc_gui import data_transfer

from pemfc_dash.main import create_settings


# from main import create_settings


def uicalc_prepare_initcalc(input_df: pd.DataFrame, i_limits: list, settings, input_cols=None) -> pd.DataFrame:
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

    data_df = create_settings(data_df, settings, input_cols=input_cols)

    data_df.loc[:, "u_pred"] = None
    data_df.loc[:, "u_pred_diff"] = None

    return data_df


def uicalc_prepare_refinement(data_df: pd.DataFrame, input_df: pd.DataFrame, settings):
    """
    ToDo: Documentation
    ToDO: Error handling: Handle None in results.
    Returns new calculation row(s)
    @return:
    """


    n = data_df.shape[0]
    new_data_df = pd.DataFrame(columns=data_df.columns)
    data_df.sort_values("simulation-current_density", ignore_index=True, inplace=True)

    if n == 3:  # first refinement, set prediction = calculation & prediction different to zero
        data_df["u_pred"] = data_df["global_data"].apply(lambda x: x["Stack Voltage"]["value"])
        data_df["u_pred_diff"] = 0
        refine = data_df

    else:  # refinement no. 2 onwards....

        # Calculate difference between calculations and prior predictions
        data_df["u_pred_diff"] = abs(data_df["global_data"].apply(lambda x: x["Stack Voltage"]["value"]) - \
                                     data_df["u_pred"])
        data_df["u_pred_diff"]= pd.to_numeric(data_df["u_pred_diff"])
        # Location of largest deviation
        idxmax = data_df["u_pred_diff"].idxmax()
        # "reset" predictions, as refinement will be performed now
        data_df.loc[idxmax, "u_pred"] = data_df.loc[idxmax, "global_data"]["Stack Voltage"]["value"]

        refine = data_df.iloc[idxmax - 1: idxmax + 2]

    # DataFrame 'refine' has 3 rows. Two additional will be added inbetween:

    i_calc = refine["simulation-current_density"].to_list()
    u_calc = refine["global_data"].apply(lambda x: x["Stack Voltage"]["value"]).to_list()

    for idx in range(len(i_calc) - 1):
        i_new = (i_calc[idx] + i_calc[idx + 1]) / 2
        u_pred = (u_calc[idx] + u_calc[idx + 1]) / 2

        new_data_df.loc[i_new, :] = data_df.iloc[0, :]  # duplicate random(here first) row and adjust current density)
        new_data_df.loc[i_new, "simulation-current_density"] = i_new
        new_data_df.loc[i_new, "u_pred"] = u_pred

    # Create settings out of (only) input columns
    new_data_df_red = new_data_df.loc[:, input_df.columns]
    # For legacy: Create "input_data"-dict, as required for data_transfer.gui_to_sim_transfer()
    new_data_df['input_data'] = new_data_df_red.apply(
        lambda row: {i: {'sim_name': i.split('-'), 'value': v} for i, v in zip(row.index, row.values)}, axis=1)

    new_data_df['settings'] = new_data_df['input_data'].apply(
        lambda x: data_transfer.gui_to_sim_transfer(x, settings)[0])

    return new_data_df

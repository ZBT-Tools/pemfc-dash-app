import pandas as pd
import numpy as np
from pemfc_gui import data_transfer




def uicalc_prepare_initcalc(input_df: pd.DataFrame, i_limits: list, settings):
    """
    Initial calculations for Ui-curve
    @return:
    """
    data_df = pd.DataFrame(columns=input_df.columns)

    # Create initial input sets
    #
    i_calc = np.linspace(i_limits[0], i_limits[1], 3)
    for i in i_calc:
        data_df.loc[i, :] = input_df.loc["nominal", :]
        data_df.loc[i, "simulation-current_density"] = float(i)

    # Create calculation df: input_data and settings columns, refinement helper columns
    new_columns = ["input_data", "settings", "u_pred", "u_pred_diff"]
    data_df.loc[:, "input_data"] = None
    data_df['input_data'] = data_df['input_data'].astype(object)
    data_df.loc[:, "settings"] = None
    data_df['settings'] = data_df['settings'].astype(object)
    data_df.loc[:, "u_pred"] = None
    data_df.loc[:, "u_pred_diff"] = None

    # For legacy: Create "input_data"-dict, as required for data_transfer.gui_to_sim_transfer()
    data_df['input_data'] = data_df.apply(
        lambda row: {i: {'sim_name': i.split('-'), 'value': v} for i, v in zip(row.index, row.values)
                     if i not in new_columns}, axis=1)

    data_df['settings'] = data_df['input_data'].apply(data_transfer.gui_to_sim_transfer, target_dict=settings)
    data_df['settings'] = data_df['settings'].apply(lambda x: x[0])

    return data_df


def uicalc_prepare_refinement(data_df,input_df,settings):
    """
    Returns new calculation row(s)
    @return:
    """
    new_columns = ["input_data", "settings", "u_pred", "u_pred_diff"]

    n = data_df.shape[0]
    new_data_df = pd.DataFrame(columns=input_df.columns)
    data_df.sort_values("simulation-current_density", ignore_index=True, inplace=True)

    if n == 3:  # first refinement
        data_df["u_pred"] = data_df["global_data"].apply(lambda x: x["Stack Voltage"]["value"])
        data_df["u_pred_diff"] = 0
        refine = data_df

    else:  # refinement no. 2 onwards....

        # Calculate difference between calculations and predictions
        data_df["u_pred_diff"] = abs(data_df["global_data"].apply(lambda x: x["Stack Voltage"]["value"]) - \
                                     data_df["u_pred"])
        idxmax = data_df["u_pred_diff"].idxmax()
        # "reset" predictions, now that calculation has been done
        data_df.loc[idxmax, "u_pred"] = data_df.loc[idxmax, "global_data"]["Stack Voltage"]["value"]

        refine = data_df.iloc[idxmax - 1: idxmax + 2]

    i_calc = refine["simulation-current_density"].to_list()
    u_calc = refine["global_data"].apply(lambda x: x["Stack Voltage"]["value"]).to_list()

    for idx in range(len(i_calc) - 1):
        i_new = (i_calc[idx] + i_calc[idx + 1]) / 2
        u_pred = (u_calc[idx] + u_calc[idx + 1]) / 2

        new_data_df.loc[i_new, :] = input_df.loc["nominal", :]
        new_data_df.loc[i_new, "simulation-current_density"] = i_new
        new_data_df.loc[i_new, "u_pred"] = u_pred

    # For legacy: Create "input_data"-dict, as required for data_transfer.gui_to_sim_transfer()
    new_data_df['input_data'] = new_data_df.apply(
        lambda row: {i: {'sim_name': i.split('-'), 'value': v} for i, v in zip(row.index, row.values)
                     if i not in new_columns}, axis=1)

    new_data_df['settings'] = new_data_df['input_data'].apply(data_transfer.gui_to_sim_transfer,
                                                              target_dict=settings)
    new_data_df['settings'] = new_data_df['settings'].apply(lambda x: x[0])

    return new_data_df

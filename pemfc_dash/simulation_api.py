import pandas as pd

from pemfc import main_app


def run_simulation(input_table: pd.DataFrame, return_unsuccessful=True) \
        -> (pd.DataFrame, bool):
    """
    API to actual simulation backend, which is here the function main in the
    pemfc module (github.com/zbt-tools/pemfc-core). The input settings are given
    in the pandas DataFrame input_table.

    - Run input_table rows, catch exceptions for single calculations
      https://stackoverflow.com/questions/22847304/exception-handling-in-pandas-apply-function
    - Append result columns to input_table
    - Return DataFrame
    """

    def func(settings):
        try:
            return main_app.main(settings)
        except Exception as E:
            return repr(E)

    settings = input_table["settings"]
    result_table = settings.progress_apply(func)
    # result_table = input_table["settings"].map(func)
    # result_table = input_table["settings"].parallel_apply(func)
    # result_table = input_table["settings"].apply_parallel(func, num_processes=4)

    input_table["global_data"] = result_table.apply(
        lambda x: x[0][0] if (isinstance(x, tuple)) else None)
    input_table["local_data"] = result_table.apply(
        lambda x: x[1][0] if (isinstance(x, tuple)) else None)
    input_table["successful_run"] = result_table.apply(
        lambda x: True if (isinstance(x[0], list)) else False)

    all_successfull = True if input_table["successful_run"].all() else False

    if not return_unsuccessful:
        input_table = input_table.loc[input_table["successful_run"], :]

    return input_table, all_successfull

import torch
import pandas as pd

from Helpers.runner import Runner, Objective, ParamsTune
from Helpers.data import DataHandler
from Helpers.plot import Plotter
from Helpers.utils import Utils

# =============================================================================
# Main Script
# =============================================================================
if __name__ == "__main__":

    u = Utils()
    seed = None
    device = u.device()

    data_name = "ECDC"
    dh = DataHandler(data_name=data_name, utils=u)
    df_full = dh.loadData()

    u.printHeaderText("countriesAndTerritories", df_full["countriesAndTerritories"].unique())
    u.printHeaderText("continentExp", df_full["continentExp"].unique())

    u.printHeader(f"Data summary for {data_name}")
    dh.summarize(df_full)

    df = dh.extendData(df=df_full)
    first_date = df['date'].min()
    last_date = df['date'].max()
    print("First date in the DataFrame:", first_date)
    print("Last date in the DataFrame:", last_date)

    filters_global = {
        'continentExp' : 'Europe'
    }
    df_eu = dh.processData(df=df, filters=filters_global)
    first_date = df_eu['date'].min()
    last_date = df_eu['date'].max()
    print("First date in the EU DataFrame:", first_date)
    print("Last date in the EU DataFrame:", last_date)

    # Group by 'countriesAndTerritories' and get the first and last date for each group
    country_dates = df.groupby('countriesAndTerritories')['date'].agg(['min', 'max'])
    # Rename columns for clarity
    country_dates = country_dates.rename(columns={'min': 'first_date', 'max': 'last_date'})
    print(country_dates)

    country_dates_eu = df_eu.groupby('countriesAndTerritories')['date'].agg(['min', 'max'])
    # Rename columns for clarity
    country_dates_eu = country_dates_eu.rename(columns={'min': 'first_date', 'max': 'last_date'})
    print(country_dates_eu)

    combined_country_dates = pd.concat([country_dates, country_dates_eu], ignore_index=True)

    # Print the combined DataFrame
    print(combined_country_dates)

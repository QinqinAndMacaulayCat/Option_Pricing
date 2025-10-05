"""
Inserts all the data from csv files in the data/ directory into a local SQLite database.
So that we can easily query the data for testing and demonstration purposes rather than loading csv files each time.
"""

import os
import sqlite3
import pandas as pd

folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"

csv_files = {
    "option_prices": folder_path + "option_prices.csv",
    "standardized_options": folder_path + "standardized_options.csv",
    "vol_surface": folder_path + "vol_surface.csv",
    "option_volume": folder_path + "option_volume.csv",
    "forward_price": folder_path + "forward_price.csv",
    "historical_vol": folder_path + "historical_vol.csv"}


conn = sqlite3.connect(folder_path + "option_data.db")
cur = conn.cursor()

for table_name, file_path in csv_files.items():
    df = pd.read_csv(file_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Inserted data into table: {table_name}")
    conn.commit()

conn.close()

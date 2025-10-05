
"""
This module fetches sample data for option pricing, which can be used for testing and demonstration purposes.
All the data is downloaded from a WRDS database.
"""
import os
import sqlite3
import pandas as pd

class SampleDataFetcher:
    folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = folder_path + '/data/option_data.db' 
    earliest_date = '2022-01-03'
    latest_date = '2023-08-31'

    def __init__(self):
        self.conn = sqlite3.connect(self.file_path)
        self.cur = self.conn.cursor()
    
    def option_prices(self, date: str,
                      ticker: str = None,
                      option_type: str = None) -> pd.DataFrame:
        """
        Fetches sample option pricing data
        
        Parameters:
        date (str): The date for which to fetch the option data in 'YYYY-MM-DD' format.

        Returns:
        pd.DataFrame: A DataFrame containing the option pricing data for the specified date.
        """

        query = f"""
        SELECT * FROM option_prices
        WHERE date = '{date}'
        AND ({'ticker = ' + repr(ticker) if ticker else '1=1'})
        AND ({'cp_flag = ' + repr(option_type) if option_type else '1=1'})
        """

        df = pd.read_sql_query(query, self.conn)

        return df
    
    def option_contracts(self, date: str,
                         ticker: str = None,
                         option_type: str = None
                         ) -> pd.DataFrame:
        """
        Fetches sample option contract data
        
        Parameters:
        date (str): The date for which to fetch the option contract data in 'YYYY-MM-DD' format.
        ticker (str, optional): The ticker symbol to filter by. Defaults to None.
        option_type (str, optional): The option type to filter by ('C' or 'P'). Defaults to None.

        Returns:
        pd.DataFrame: A DataFrame containing the option contract data for the specified date.
        """
        query = f"""
        SELECT * FROM standardized_options
        WHERE date = '{date}'
        AND ({'ticker = ' + repr(ticker) if ticker else '1=1'})
        AND ({'cp_flag = ' + repr(option_type) if option_type else '1=1'})
        """

        df = pd.read_sql_query(query, self.conn)

        return df

    def vol_surface(self, date: str,
                    ticker: str = None,
                    option_type: str = None
                    ) -> pd.DataFrame:
        """
        Fetches sample volatility surface data
        
        Parameters:
        date (str): The date for which to fetch the volatility surface data in 'YYYY-MM-DD' format.
        ticker (str, optional): The ticker symbol to filter by. Defaults to None.
        option_type (str, optional): The option type to filter by ('C' or 'P'). Defaults to None.

        Returns:
        pd.DataFrame: A DataFrame containing the volatility surface data for the specified date.
        """
        query = f"""
        SELECT * FROM vol_surface
        WHERE date = '{date}'
        AND ({'ticker = ' + repr(ticker) if ticker else '1=1'})
        AND ({'cp_flag = ' + repr(option_type) if option_type else '1=1'})
        """

        df = pd.read_sql_query(query, self.conn)
        return df
    
    def forward_prices(self, date: str,
                       ticker: str = None,
                       ) -> pd.DataFrame:
        
        """
        Fetches sample forward price data
        Parameters:
        date (str): The date for which to fetch the forward price data in 'YYYY-MM-DD' format.
        ticker (str, optional): The ticker symbol to filter by. Defaults to None.
        option_type (str, optional): The option type to filter by ('C' or 'P    '). Defaults to None.

        Returns:
        pd.DataFrame: A DataFrame containing the forward price data for the specified date.
        """
        query = f"""
        SELECT * FROM forward_price
        WHERE date = '{date}'
        AND ({'ticker = ' + repr(ticker) if ticker else '1=1'})
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def option_volume(self, date: str,
                      ticker: str = None,
                      option_type: str = None
                      ) -> pd.DataFrame:
        """
        Fetches sample option volume data
        Parameters:
        date (str): The date for which to fetch the option volume data in 'YYYY-MM-DD' format.
        ticker (str, optional): The ticker symbol to filter by. Defaults to None.
        option_type (str, optional): The option type to filter by ('C' or 'P'). Defaults to None.
        Returns:
        pd.DataFrame: A DataFrame containing the option volume data for the specified date.
        """
        query = f"""
        SELECT * FROM option_volume
        WHERE date = '{date}'
        AND ({'ticker = ' + repr(ticker) if ticker else '1=1'})
        AND ({'cp_flag = ' + repr(option_type) if option_type else '1=1'})
        """
        df = pd.read_sql_query(query, self.conn)
        return df



    def close(self):
        self.cur.close()
        self.conn.close()

import pandas as pd

class DataManager:
    _instance = None

    def __init__(self):
        self._df = None
        self._filename = ""

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, dataframe: pd.DataFrame):
        self._df = dataframe

    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, name: str):
        self._filename = name

    def set_data(self, dataframe: pd.DataFrame, filepath: str = ""):
        """Set both dataframe and filename at once."""
        self._df = dataframe
        self._filename = filepath

    def get_column(self, col_name):
        if self._df is not None and col_name in self._df.columns:
            return self._df[col_name]
        return None
    
    def get_columns(self):
        if self._df is not None:
            return list(self._df.columns)
        return []

    def get_data_matrix(self, col_names):
        """Returns numpy array for specified columns"""
        if self._df is not None:
            return self._df[col_names].to_numpy(dtype=float)
        return None

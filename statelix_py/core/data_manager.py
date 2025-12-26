import pandas as pd
import numpy as np

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

    def load_csv(self, filepath: str):
        """Load data from a CSV file with auto-encoding detection."""
        # Try multiple encodings for Japanese CSV files
        encodings = ['utf-8', 'cp932', 'shift_jis', 'utf-8-sig', 'latin1']
        last_error = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                
                # Check if first row looks like Japanese labels (SSDSE-style)
                # If all values in first row are non-numeric strings, skip it
                first_row = df.iloc[0] if len(df) > 0 else None
                if first_row is not None:
                    non_numeric_count = 0
                    for val in first_row.values[:10]:  # Check first 10 cols
                        try:
                            float(str(val).replace(',', ''))
                        except (ValueError, TypeError):
                            non_numeric_count += 1
                    
                    if non_numeric_count >= 7:  # Likely a label row
                        df = df.iloc[1:].reset_index(drop=True)
                
                # Auto-convert numeric columns
                for col in df.columns:
                    try:
                        # Try to convert to numeric, handling comma separators
                        converted = pd.to_numeric(
                            df[col].astype(str).str.replace(',', ''), 
                            errors='coerce'
                        )
                        # If more than 50% are valid numbers, use the conversion
                        if converted.notna().sum() > len(df) * 0.5:
                            df[col] = converted
                    except Exception:
                        pass
                
                self._df = df
                self._filename = filepath
                self._history = []
                self._value_labels = {}
                return True
            except UnicodeDecodeError:
                last_error = f"Encoding {encoding} failed"
                continue
            except Exception as e:
                last_error = str(e)
                continue
        
        print(f"Error loading CSV: {last_error}")
        return False

    def set_data(self, dataframe: pd.DataFrame, filepath: str = ""):
        """Set both dataframe and filename at once."""
        self._df = dataframe
        self._filename = filepath
        self._history = []
        self._value_labels = {}

    def log_operation(self, code: str):
        if not hasattr(self, '_history'): self._history = []
        self._history.append(code)

    def get_history(self):
        return getattr(self, '_history', [])

    def impute_missing(self, col_name, strategy, fill_value=None):
        if self._df is None or col_name not in self._df.columns: return False
        
        if strategy == 'mean':
            val = self._df[col_name].mean()
        elif strategy == 'median':
            val = self._df[col_name].median()
        elif strategy == 'mode':
            val = self._df[col_name].mode()[0]
        elif strategy == 'constant':
            val = fill_value
        else:
            return False

        self._df[col_name] = self._df[col_name].fillna(val)
        self.log_operation(f"df['{col_name}'] = df['{col_name}'].fillna({repr(val)})")
        return True

    def encode_categorical(self, col_name, method='one-hot'):
        if self._df is None or col_name not in self._df.columns: return False
        
        if method == 'one-hot':
            new_cols = pd.get_dummies(self._df[col_name], prefix=col_name)
            self._df = pd.concat([self._df, new_cols], axis=1)
            self.log_operation(f"df = pd.concat([df, pd.get_dummies(df['{col_name}'], prefix='{col_name}')], axis=1)")
        elif method == 'label':
            self._df[col_name] = self._df[col_name].astype('category').cat.codes
            self.log_operation(f"df['{col_name}'] = df['{col_name}'].astype('category').cat.codes")
        else:
            return False
        return True

    def convert_type(self, col_name, new_type):
        if self._df is None or col_name not in self._df.columns: return False
        try:
            self._df[col_name] = self._df[col_name].astype(new_type)
            self.log_operation(f"df['{col_name}'] = df['{col_name}'].astype('{new_type}')")
            return True
        except (ValueError, TypeError):
            return False

    def set_value_labels(self, col_name, mapping: dict):
        if not hasattr(self, '_value_labels'): self._value_labels = {}
        self._value_labels[col_name] = mapping

    def get_value_labels(self, col_name):
        return getattr(self, '_value_labels', {}).get(col_name, None)

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
            # Ensure we only use numeric columns for the matrix
            numeric_df = self._df[col_names].select_dtypes(include=[np.number])
            return numeric_df.to_numpy(dtype=float)
        return None

    def merge_data(self, other_df: pd.DataFrame, on: str, how: str = 'inner'):
        if self._df is None: return False
        try:
            self._df = pd.merge(self._df, other_df, on=on, how=how)
            self.log_operation(f"df = pd.merge(df, other_df, on='{on}', how='{how}')")
            return True
        except Exception as e:
            print(f"Merge error: {e}")
            return False

    def reshape_data(self, method, **kwargs):
        if self._df is None: return False
        try:
            if method == 'melt':
                # kwargs: id_vars, value_vars
                self._df = pd.melt(self._df, **kwargs)
                self.log_operation(f"df = pd.melt(df, **{repr(kwargs)})")
            elif method == 'pivot':
                # kwargs: index, columns, values
                self._df = self._df.pivot(**kwargs).reset_index()
                self.log_operation(f"df = df.pivot(**{repr(kwargs)}).reset_index()")
            else:
                return False
            return True
        except Exception as e:
            print(f"Reshape error: {e}")
            return False

    # --- Survey Weights ---
    def set_weight_column(self, col_name: str):
        if self._df is None or col_name not in self._df.columns: return False
        if not hasattr(self, '_weight_col'): self._weight_col = None
        self._weight_col = col_name
        self.log_operation(f"# Set weight column: {col_name}")
        return True

    def get_weight_column(self):
        return getattr(self, '_weight_col', None)

    def get_weighted_mean(self, col_name: str) -> float:
        if self._df is None or col_name not in self._df.columns: return np.nan
        w_col = self.get_weight_column()
        if w_col and w_col in self._df.columns:
            weights = self._df[w_col].values
            values = self._df[col_name].values
            return np.average(values, weights=weights)
        return self._df[col_name].mean()

    def get_weighted_std(self, col_name: str) -> float:
        if self._df is None or col_name not in self._df.columns: return np.nan
        w_col = self.get_weight_column()
        if w_col and w_col in self._df.columns:
            weights = self._df[w_col].values
            values = self._df[col_name].values
            avg = np.average(values, weights=weights)
            variance = np.average((values - avg) ** 2, weights=weights)
            return np.sqrt(variance)
        return self._df[col_name].std()

    # --- Multiple Imputation (MICE) ---
    def impute_mice(self, n_imputations: int = 5, max_iter: int = 10):
        if self._df is None: return False
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            
            num_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols: return False
            
            imputer = IterativeImputer(max_iter=max_iter, random_state=0)
            self._df[num_cols] = imputer.fit_transform(self._df[num_cols])
            
            self.log_operation(f"# MICE imputation (n={n_imputations}, max_iter={max_iter})\n"
                             f"from sklearn.impute import IterativeImputer\n"
                             f"imputer = IterativeImputer(max_iter={max_iter})\n"
                             f"df[{repr(num_cols)}] = imputer.fit_transform(df[{repr(num_cols)}])")
            return True
        except Exception as e:
            print(f"MICE imputation error: {e}")
            return False

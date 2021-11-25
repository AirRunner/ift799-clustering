import pandas as pd


def prepare_data():
    # Load data
    df = pd.read_csv("data/res_2000.csv", index_col='x')
    df.index = pd.to_datetime(df.index)

    # Normalize series
    df_norm = (df - df.mean()) / df.std()

    return df_norm

def create_windows(df, window_size, window_shift=0):
    if window_shift < 1:
        window_shift = window_size

    dates = []
    windows = []
    i = 0
    
    while i <= df.shape[0] - window_size:
        w = df.iloc[i:i + window_size]
        dates.append(w.index[0])
        windows.append(w.values.T)
        
        i += window_shift
    
    return dates, windows

import pandas as pd

def load_spambase_dataset():
    """
    Loads the Spambase dataset from the raw folder.
    """
    path = 'data/raw/spambase.data'
    df = pd.read_csv(path, header=None)
    return df

import pandas as pd
from SLP import Perceptron
from adaline import Adaline


def get_data_path():
    return 'processed_data/processed_data.csv'

def concate_data_frames(df0,df1):
    result = pd.concat([df0, df1], ignore_index=True)
    return result


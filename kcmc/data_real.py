import numpy as np
import pandas as pd
import torch
from .data_binary import estimate_p_t

def load_data():
    df = pd.read_csv("union1978.csv")
    df.columns = ("id", "age", "black", "educ76", "smsa", "south", "married", "enrolled",
                     "educ78", "manufacturing", "occupation", "union", "wage")

    df.age = df.age + 12  # age was recorded in 1966
    df["education"] = np.maximum(df.educ76, df.educ78)
    df.black = (df.black == 2).astype(int)
    df.married = np.logical_or(df.married == 1, df.married == 2).astype(int)
    df.smsa = np.logical_or(df.smsa == 1, df.smsa == 2).astype(int)
    df.manufacturing = np.logical_and(206 <= df.manufacturing, df.manufacturing <= 459).astype(int)

    def get_occupation_id(occ_number):
        if 401 <= occ_number <= 545:
            return 0  # craftsman
        elif 960 <= occ_number <= 985:
            return 1  # laborer
        else:
            return 2  # other
        
    df.occupation = df.occupation.apply(get_occupation_id)

    df = df.drop(columns=['id', 'educ76', 'educ78', 'enrolled'])
    df = df[df.occupation != 2]

    # remove missing values
    missing = np.logical_or(df.to_numpy() == -4, df.to_numpy() == -5)
    df = df[~missing.any(axis=1)]

    # df.describe()

    Y = df.wage.to_numpy()
    Y = torch.as_tensor(Y)
    T = df.union.to_numpy()
    X = df.drop(columns=['wage', 'union']).to_numpy()

    return Y, T, X


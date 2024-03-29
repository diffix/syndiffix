import pandas as pd
import numpy as np
import os
from pandas.errors import ParserError

from syndiffix import Synthesizer

def count_identical_rows(df1, df2):
    # Merge the two dataframes on all columns
    merged_df = pd.merge(df1, df2, how='inner')
    
    # The number of identical rows is the number of rows in the merged dataframe
    num_identical_rows = len(merged_df)
    
    return num_identical_rows

def test2():
    catCols = [ 'SEX', 'MSP', 'HISP', 'RAC1P', 'HOUSING_TYPE', 
              'OWN_RENT', 'INDP_CAT', 'EDU', 'PINCP_DECILE',
              'DVET', 'DREM', 'DEYE', 'DEAR', 'DPHY', 
              ]
    csv_path = os.path.join('c:\\', 'paul', 'sdnist', 'diverse_communities_data_excerpts', 'texas', 'tx2019.csv')
    print(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    for col in catCols:
        is_numeric = np.issubdtype(df[col].dtypes, np.number)
        print(f"{col} is numeric {is_numeric}")
        print(df[col].unique())
        null_values = df[col].isnull()
        # To count the number of null values
        print(f"     {null_values.sum()} null values")
    df = df.sample(n=1000)
    # HISP and SEX are numeric, so let's change only HISP to string
    df['HISP'] = df['HISP'].astype(str)
    synth = Synthesizer(df[['HISP','SEX']])
    df_syn = synth.sample()
    print(df_syn.dtypes)
    print(df_syn.head())
    pass

def test1():
    csv_path = os.path.join('c:\\', 'paul', 'datasets', 'banking.loans', 'original', 'loan_account_card_clients.csv')
    print(csv_path)
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""], low_memory=False)
    # Try to infer datetime columns.
    for col in df.columns[df.dtypes == "object"]:
        try:
            df[col] = pd.to_datetime(df[col], format="ISO8601")
        except (ParserError, ValueError):
            pass

    print("make synthesizer")
    synthesizer = Synthesizer(df)
    print("first sample")
    df_syn1 = synthesizer.sample()
    print(df_syn1[0:5].to_string())
    print("second sample")
    df_syn2 = synthesizer.sample()
    print(df_syn2[0:5].to_string())

    cnt = count_identical_rows(df_syn1, df_syn2)
    print(f"There are {cnt} identical rows")

if False:
    test1()
if True:
    test2()
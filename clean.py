#Importing
import sklearn
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import os

def remove_outliers(df):
    # Load the dataset
    data_to_use = df.copy()

    #print("Old Shape: ", data_to_use.shape)

    ''' Detection '''
    # IQR
    # Calculate the upper and lower limits
    Q1 = data_to_use['fare_amount'].quantile(0.25)
    Q3 = data_to_use['fare_amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(data_to_use['fare_amount']>=upper)[0]
    lower_array = np.where(data_to_use['fare_amount']<=lower)[0]

    # Removing the outliers
    data_to_use.drop(index=upper_array, inplace=True)
    data_to_use.drop(index=lower_array, inplace=True)

    # Print the new shape of the DataFrame
    #print("New Shape: ", data_to_use.shape)

    # Uncomment to verify correct transformation:
    #NOTEBOOKPATH = "/content/drive/MyDrive/cs370/"
    #data_to_use.to_csv(os.path.join(NOTEBOOKPATH, "no_outlier_taxi_data.csv"))
    print(f"The old max for fare amount is {df['fare_amount'].max()}")
    print(f"The new max for fare amount is {data_to_use['fare_amount'].max()}")
    print()
    return data_to_use

def remove_negatives(df):
    #print("Old Shape: ", df.shape)
    df_new = df.loc[df['fare_amount'] >= 0]
    #print("New Shape: ", df_new.shape)
    print(f"The old min for fare amount is {df['fare_amount'].min()}")
    print(f"The new min for fare amount is {df_new['fare_amount'].min()}")
    return df_new


def clean_the_data(df):
  print("Old Shape: ", df.shape)
  print(f"The original train max for fare amount is {df['fare_amount'].max()}")
  print(f"The original train min for fare amount is {df['fare_amount'].min()}")
  no_fare_outliers = remove_outliers(df)
  no_fare_outliers_or_negatives = remove_negatives(no_fare_outliers)

  print()
  print("New Shape: ", no_fare_outliers_or_negatives.shape)
  print(f"The new train max for fare amount is {no_fare_outliers_or_negatives['fare_amount'].max()}")
  print(f"The new train min for fare amount is {no_fare_outliers_or_negatives['fare_amount'].min()}")

  return no_fare_outliers_or_negatives

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import holidays
import numpy as np
import datetime
import pandas as pd

def trip_duration(df):
  dt6 = []
  for index, row in df.iterrows():
    pick_up = datetime.datetime.strptime(row['tpep_pickup_datetime'],  "%m/%d/%Y %I:%M:%S %p")
    drop_off = datetime.datetime.strptime(row['tpep_dropoff_datetime'],  "%m/%d/%Y %I:%M:%S %p")
    trip_duration = drop_off - pick_up
    trip_duration = trip_duration.total_seconds()
    dt6.append(trip_duration)
  df['trip_duration'] = dt6
  return df


def day_of_week(df):
  dt_weekday = []
  for i in df['tpep_pickup_datetime']:
    dt1 = datetime.datetime.strptime(i, "%m/%d/%Y %I:%M:%S %p").strftime('%w')
    dt_weekday.append(dt1)
  df['day_of_week'] = dt_weekday
  print('Unique counts of week:', df['day_of_week'].unique())
  return df

def is_holiday(df):
  us_holidays = holidays.UnitedStates()
  holiday = []
  for i in df['tpep_pickup_datetime']:
    dt = datetime.datetime.strptime(i, "%m/%d/%Y %I:%M:%S %p")
    is_hol = dt in us_holidays
    holiday.append(is_hol)
  df['is_holiday'] = holiday
  df['is_holiday'] =  df['is_holiday'].astype(np.int64)
  return df

def transformTaxi(df):
    ''' Transform the key features of the taxi dataset for model fitting.
        Save the transformed data in "data/transformedTaxi.csv"
    '''
    print(df[df['fare_amount'].isnull()])
    y = df['fare_amount']

    num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler())
        ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

    num_attribs = ["trip_distance", "trip_duration"]
    cat_attribs = ["PULocationID", "DOLocationID", "day_of_week", 'is_holiday', "passenger_count", "payment_type"]
    #do we want anything else to be categorical vs numeric

    print(df[num_attribs + cat_attribs + ["fare_amount"] ].head())


    colTransformer = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ], verbose = True)

    df = df[num_attribs + cat_attribs]
    X = colTransformer.fit_transform(df)

    print("Original column headers:", colTransformer.feature_names_in_)
    print("Col headers after transformation:", colTransformer.get_feature_names_out())

    # Uncomment to verify correct transformation:
    newdf = pd.DataFrame(X, columns = colTransformer.get_feature_names_out())
    newdf['fare_amount'] = y
    print(newdf.head())
    #newdf.to_csv(os.path.join(notebookpath, "Train and Test Data/TransformedData.csv"))


    return X, y, newdf


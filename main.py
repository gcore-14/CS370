import explore
import transformAndScale as tAS
import pandas as pd
import clean
import trainAndEvaluate

#Read in train data
df = pd.read_csv("trainData.csv")
#Remove index column
df = df[df.columns[1:]]
#Explore Data
#explore.exploredata(df)

#Cleans the data by removing outliers outside of IQR range, as well as negative fare values for taxi rides.
df = clean.clean_the_data(df)
print(df[df['fare_amount'].isnull()])

#Creates Trip Duration Column using Pickup Time and Dropoff Time
df = tAS.trip_duration(df)

#Creates a new column of what day of the week it is based on pickup time
df = tAS.day_of_week(df)

#Creates a new column for if the taxi ride occoured on a US calendar holiday
df = tAS.is_holiday(df)

#Transforms the taxi data by applying a Pipeline for numerical values that uses a simple scaler and replaces with mean values.
#The Categorical Pipeline uses one-hot encoding for all categorical data, and replaces missing values with the most
#frequent values. There was no missing values for either pipeline, but we added them incase future data requires them.
X, y, df = tAS.transformTaxi(df)


trainAndEvaluate.trainAllModels(df)




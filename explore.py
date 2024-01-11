import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def exploredata(df):
    ''' for dataframe df creates a file with columns corresponding to cols in df,
    but values in the columns are just distiinct values in df'''
    df2 = pd.DataFrame(columns= df.columns, index = range(len(df) + 1))
    #df2 = df2.reindex()
    for col in df.columns:
        num = len(df[col].value_counts().index)
        df2.loc[0, col] = num
        df2.loc[1: num, col] = df[col].value_counts().index
    print("Dataframe of Unique Values: ")
    print(df2)
    print('-'*25,'Data Characteristics:', '-'*25)
    print('-'*30 + "Shape and Header" + '-'*30)
    print (df.shape, df.head())
    print('-'*30 + "Tail" + '-'*30)
    print ( df.tail())
    print('-'*30 + "Info" + '-'*30)
    df.info()
    print('-'*30 + "Describe" + '-'*30)
    print(df.describe())
    print('-'*65)
    print('Column            Num unique values')
    print(df.nunique())

    df2 = df[['VendorID', 'passenger_count', "tip_amount", 'tolls_amount']]

    df.iloc[:,1:].hist(bins=50, figsize=(12, 8))
    plt.show()

    fig, axes = plt.subplots(3,3)
    #class0 = df[df[targetCol] == 0]
    #class1 = df[df[targetCol] == 1]

    ax = axes.ravel()
    #new_sex = pd.get_dummies(df, columns=['Sex'])
    #print(new_sex)

    targetCol = "fare_amount"
    for i in range(len(df2.columns)):
        _, bins = np.histogram(df2.iloc[:, i], bins=50)
        ax[i].hist(df.iloc[:, i], bins=bins, color='b', alpha=.5)
        #ax[i].hist(df.iloc[:, i], bins=bins, color='g', alpha=.5)
        ax[i].set_title(df2.columns[i])
        #ax[i].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend([targetCol + ' is 0', targetCol+ ' is 1'], loc="best")
    fig.tight_layout()

    plt.show()




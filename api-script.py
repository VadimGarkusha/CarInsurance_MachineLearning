import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Read-in train and test datasets
#train = pd.read_csv('/Users/13hea/Desktop/CarInsurance_MachineLearning/data/carInsurance_train.csv')
pd.set_option('display.max_columns',30) # set the maximum width

# Load the dataset in a dataframe object

df = pd.read_csv('/Users/13hea/Desktop/CarInsurance_MachineLearning/data/carInsurance_train.csv')

# Explore the data check the column values
print(df.columns.values)
print (df.head())
categories = []
for col, col_type in df.dtypes.iteritems():
    if col_type == 'O':
        categories.append(col)
    else:
        df[col].fillna(0, inplace=True)
        
print(categories)
print(df.columns.values)
print(df.head())
df.describe()
df.dtypes
#check for null values
print(len(df) - df.count()) #Cabin , boat, home.dest have so many missing values


include = ['Age', 'Marital', 'Outcome']
df_ = df[include]
print(df_.columns.values)
print(df_.head())
df_.describe()
df_.dtypes
df_['Marital'].unique()
df_['Age'].unique()
#df_['Outcome'].unique()

# check the null values
print(df_.isnull().sum())
print(df_['Marital'].isnull().sum())
print(len(df_) - df_.count())

df_.dropna(axis=0,how='any',inplace=True)


categoricals = []

for col, col_type in df_.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace=True)
print(categoricals)


df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=False)
pd.set_option('display.max_columns',30)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())


from sklearn import preprocessing

# Get column names first
names = df_ohe.columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df['Age'].describe())
print(scaled_df['Marital_single'].describe())
print(scaled_df['Marital_married'].describe())
print(scaled_df['Marital_divorced'].describe())
#print(scaled_df['Outcome_success'].describe())
#print(scaled_df['Outcome_failure'].describe())
#print(scaled_df['Outcome_NA'].describe())
#print(scaled_df['embarked_C'].describe())
#print(scaled_df['embarked_Q'].describe())
#print(scaled_df['embarked_S'].describe())
#print(scaled_df['survived'].describe())
print(scaled_df.dtypes)


dependent_variable = 'Age'

# Another way to split the three features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = scaled_df[dependent_variable]

#convert the class back into integer
y = y.astype(int)

# Split the data into train test
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)

#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)

# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)



testY_predict = lr.predict(testX)

testY_predict.dtype

#print(testY_predict)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))

#Let us print the confusion matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))

import joblib
joblib.dump(lr, '/Users/13hea/Desktop/CarInsurance_MachineLearning/data/model_lr2.pkl')
print("Model dumped!")

model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, '/Users/13hea/Desktop/CarInsurance_MachineLearning/data/model_columns.pkl')
print("Models columns dumped!")

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            print(model_columns)
            print(lr)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to titanic model APIs!"

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('/Users/13hea/Desktop/CarInsurance_MachineLearning/data/model_lr2.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('/Users/13hea/Desktop/CarInsurance_MachineLearning/data/model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
   
    app.run(port=port, debug=True)
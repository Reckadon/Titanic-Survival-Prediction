# 1st prediction with tree model and (Sex_male	Sex_female	is_child	p_class) features

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

import os
print(os.getcwd())
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder

def process_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    processed_data = pd.DataFrame(data['PassengerId'])

    # dummy variable encoding for Sex of passenger
    encoder = OneHotEncoder(categories=[['male', 'female']], sparse_output=False).set_output(transform = 'pandas')
    onehot_sex_data = encoder.fit_transform(data[['Sex']])
    processed_data = pd.concat([processed_data, onehot_sex_data], axis=1)  
    
    # creating column for child
    age_threshold = 15
    processed_data['is_child'] = (data['Age'] < age_threshold).astype(float)
    # removing overlap from Sex_male and Sex_female columns
    processed_data.loc[processed_data['is_child'] == 1, ['Sex_male', 'Sex_female']] = 0

    # adding pclass - economic status
    processed_data['p_class'] = data['Pclass'].astype(int)
    print("Made dataset with columns:", processed_data.columns, "and length:", len(processed_data))
    return processed_data


training_filename = './data/train.csv' # local
# with open(training_filename) as:

train_set = pd.read_csv(training_filename)

processed_train_data = process_dataframe(train_set)

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(processed_train_data, train_set['Survived'])

testing_filename = './data/test.csv' # local
test_set = pd.read_csv(testing_filename)

processed_test_data = process_dataframe(test_set)

predictions = tree_model.predict(processed_test_data)
prediction_df = pd.DataFrame(test_set['PassengerId'])
prediction_df['Survived'] = predictions

submission_filename = "./predictions/TreeModelSubmissionWithOverlapRemoved.csv"
prediction_df.to_csv(submission_filename, index=False)
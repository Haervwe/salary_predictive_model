from src import data_loading
from src import preprocessing
from src import feature_engenieering
from src import modeling
from src import evaluation

#files path for the raw dataset:
data_files = ['./data/people.csv','./data/descriptions.csv','./data/salary.csv',]

#merge datasets in a cohesive Dataframe
full_dataset = data_loading.load_data(data_files)

#preprocessing of the dataframe adds missing values with LLM inference over descriptions of each row, drops the incomplete rows and cleans up the data.
cleansed_dataset = preprocessing.preprocess(full_dataset)

#split the dataset into an 80 / 20 ratio for training and testing.
X_train, X_test, y_train, y_test = feature_engenieering.split_data(cleansed_dataset)


#normalize and scale the datasets using MinMaxScaler and target encoder

normalized_X_train, te, scaler = feature_engenieering.normalize_train_data(X_train, y_train)

normalized_X_test = feature_engenieering.normalize_test_data(X_test, te, scaler)

#train the model using a random forest regressor algorithm and print out the predictions for the normalized test data.

model = modeling.train_model(normalized_X_train, y_train)

#use the test dataset to predict salaries based on the trained model.




print(evaluation.evaluate_model(normalized_X_test, y_test, model))

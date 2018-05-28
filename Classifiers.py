import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn import svm, preprocessing
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_selection
from sklearn import linear_model


def load_data(file_name):
    data = pd.read_csv('kickstarter_data_cleaned.csv')

    data = data.drop(columns=['name', 'deadline', 'launched', 'pledged', 'backers', 'usd pledged', 'usd_pledged_real',
                              'pledged_minus_goal', 'average_pledge_usd'])

    return data


# Encodes categorical variables
def process_catg_vars(data):

    # Main Categories
    m_category_encoder = preprocessing.LabelEncoder()
    unique_m_categories = data.main_category.unique()
    m_category_encoder.fit(unique_m_categories)
    data['main_category'] = m_category_encoder.transform(data['main_category'])

    # Categories
    category_encoder = preprocessing.LabelEncoder()
    unique_categories = data.category.unique()
    category_encoder.fit(unique_categories)
    data['category'] = category_encoder.transform(data['category'])

    # Currency
    currency_encoder = preprocessing.LabelEncoder()
    unique_currencies = data.currency.unique()
    currency_encoder.fit(unique_currencies)
    data['currency'] = currency_encoder.transform(data['currency'])

    # Country
    country_encoder = preprocessing.LabelEncoder()
    unique_countries = data.country.unique()
    country_encoder.fit(unique_countries)
    data['country'] = country_encoder.transform(data['country'])

    # State
    state_encoder = preprocessing.LabelEncoder()
    unique_states = data.state.unique()
    state_encoder.fit(unique_states)
    data['state'] = state_encoder.transform(data['state'])

    data = data.dropna()

    return data


# Trains the model:
def train(features_only, dep_var_only, model_name):

    if model_name == "svm":
        model = svm.SVC(random_state=10, kernel='rbf')
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=10)
    elif model_name == "bernoulli":
        model = BernoulliNB()
    elif model_name == "linear":
        model = linear_model.LinearRegression()


    # Scales the data
    features_only = preprocessing.scale(features_only)

    features_train, features_test, target_train, target_test = train_test_split(features_only, dep_var_only,
                                                                                test_size=0.2, random_state=10,
                                                                                shuffle=True)


    # Get mutual information:
    scores = feature_selection.mutual_info_classif(features_train, target_train, random_state=10)

    # Most important: Goal amount, then sub-category and duration,
    # Least important: Country and currency
    print(scores)

    # Cs = [2000, 10000, 12000, 12200, 12250, 12300, 12400, 15000]
    n_estimators = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000] ## TODO
    max_features = ['auto', 'sqrt'] ## TODO
    param_grid = {'max_features': max_features, 'n_estimators': n_estimators} ## TODO
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    #
    grid_search.fit(features_train, target_train)
    print(grid_search.best_params_)

    grid_search.fit(features_train, target_train)
    # No parameter tuning, accuracy: 0.8

    # best_params = ?

    # best_grid = grid_search.best_estimator_
    target_pred = grid_search.predict(features_test)

    return target_pred, target_test, grid_search


if __name__ == "__main__":
    # Load data from CSV:
    train_data = load_data('train.csv')

    # Data for one category
    train_data = train_data[train_data['main_category'] == 'Dance']

    # Converts categorical variables
    train_data = process_catg_vars(train_data)

    # Separates data into features and dependent variables
    dep_var = list(train_data['state'])
    features = train_data.drop(columns=['state']).as_matrix()

    # Train and test model:
    pred, test, model = train(features, dep_var, "random_forest")
    print("Model Accuracy: " + str(accuracy_score(test, pred)))
    print("Recall: " + str(recall_score(test, pred)))
    print("F1 score: " + str(f1_score(test, pred)))
    print("Precision: " + str(precision_score(test, pred)))
    #
    # test_data = INPUT DATA FROM USER
    # test_data = preprocessing.scale(test_data, axis=1)
    # test_data = pd.DataFrame(test_data)
    # results = model.predict(test_data)
    # DO SOMETHING WITH RESULTS
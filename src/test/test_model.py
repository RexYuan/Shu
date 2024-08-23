from fairness_checker import *

import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("compas_train.csv")
df = df.loc[:, ['sex', 'age_cat', 'race', 'priors_count', 'c_charge_degree', 'two_year_recid', 'decile_score']]

sex_encoder = preprocessing.LabelEncoder()
age_encoder = preprocessing.LabelEncoder()
race_encoder = preprocessing.LabelEncoder()
degree_encoder = preprocessing.LabelEncoder()

sex_cat = sex_encoder.fit_transform(df['sex'])
age_cat = age_encoder.fit_transform(df['age_cat'])
race_cat = race_encoder.fit_transform(df['race'])
degree_cat = degree_encoder.fit_transform(df['c_charge_degree'])

sex_lookup = dict(zip(sex_encoder.classes_, sex_encoder.transform(sex_encoder.classes_)))
age_lookup = dict(zip(age_encoder.classes_, age_encoder.transform(age_encoder.classes_)))
race_lookup = dict(zip(race_encoder.classes_, race_encoder.transform(race_encoder.classes_)))
degree_lookup = dict(zip(degree_encoder.classes_, degree_encoder.transform(degree_encoder.classes_)))

df['sex_cat'] = sex_cat
df['age_cat'] = age_cat
df['race_cat'] = race_cat
df['degree_cat'] = degree_cat

dummy_fields = ['sex', 'age_cat', 'race', 'c_charge_degree']
data = df.drop(dummy_fields, axis = 1)

data = df.reindex(['sex_cat', 'age_cat', 'race_cat', 'degree_cat', 'two_year_recid'], axis=1)
X_train = data.iloc[:, 0:4]
Y_train = data.iloc[:, 4]

model = LogisticRegression()
model.fit(X_train, Y_train)

data = df.reindex(['sex_cat', 'age_cat', 'race_cat', 'degree_cat', 'decile_score'], axis=1)
X_train = data.iloc[:, 0:4]
Y_train = data.iloc[:, 4]

tree_model = RandomForestClassifier()
tree_model.fit(X_train, Y_train)

df = pd.read_csv("compas_test.csv")
df = df.loc[:, ['sex', 'age_cat', 'race', 'priors_count', 'c_charge_degree', 'two_year_recid', 'decile_score']]

sex_cat = sex_encoder.fit_transform(df['sex'])
age_cat = age_encoder.fit_transform(df['age_cat'])
race_cat = race_encoder.fit_transform(df['race'])
degree_cat = degree_encoder.fit_transform(df['c_charge_degree'])

df['sex_cat'] = sex_cat
df['age_cat'] = age_cat
df['race_cat'] = race_cat
df['degree_cat'] = degree_cat

dummy_fields = ['sex', 'age_cat', 'race', 'c_charge_degree']
data = df.drop(dummy_fields, axis = 1)

data = df.reindex(['sex_cat', 'age_cat', 'race_cat', 'degree_cat', 'two_year_recid'], axis=1)
X_test = data.iloc[:, 0:4]
Y_test = data.iloc[:, 4]
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Logistic Accuracy:", accuracy)

data = df.reindex(['sex_cat', 'age_cat', 'race_cat', 'degree_cat', 'decile_score'], axis=1)
X_test = data.iloc[:, 0:4]
Y_test = data.iloc[:, 4]
Y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Deicision Tree Accuracy:", accuracy)

class compas_model_wrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, file):
        df = pd.read_csv(file)
        X = self.preprocessing(df)

        return self.model.predict(X)

    def preprocessing(self, df):
        df = df.loc[:, ['sex', 'age_cat', 'race', 'priors_count', 'c_charge_degree']]

        sex_cat = sex_encoder.fit_transform(df['sex'])
        age_cat = age_encoder.fit_transform(df['age_cat'])
        race_cat = race_encoder.fit_transform(df['race'])
        degree_cat = degree_encoder.fit_transform(df['c_charge_degree'])

        df['sex_cat'] = sex_cat
        df['age_cat'] = age_cat
        df['race_cat'] = race_cat
        df['degree_cat'] = degree_cat

        dummy_fields = ['sex', 'age_cat', 'race', 'c_charge_degree']
        data = df.drop(dummy_fields, axis = 1)

        data = df.reindex(['sex_cat', 'age_cat', 'race_cat', 'degree_cat'], axis=1)
        return data

trained = compas_model_wrapper(model)
trained_tree = compas_model_wrapper(tree_model)

c = fairness_model_checker('compas_score.csv')
race = 'African-American'
def pos_pred(Y: int) -> bool:
    return Y == 1
def calib_pred(u: int, l: int):
    def pred(Y: int) -> bool:
        return l <= int(Y) and int(Y) <= u
    return pred
def sco_pred(Y: int) -> int:
    return int(Y)
c.disparate_impact(0.8, trained, lambda row: row['race'] != race, pos_pred)
c.demographic_parity(0.2, trained, lambda row: row['race'] != race, pos_pred)
c.equalized_odds(0.2, trained, lambda row: row['race'] != race, pos_pred, lambda row: row['two_year_recid'] == '1')
c.equal_opportunity(0.2, trained, lambda row: row['race'] != race, pos_pred, lambda row: row['two_year_recid'] == '1')
c.accuracy_eqaulity(0.2, trained, lambda row: row['race'] != race, pos_pred, lambda row: row['is_recid'] == '1')
c.predictive_parity(0.2, trained, lambda row: row['race'] != race, pos_pred, lambda row: row['is_recid'] == '1')
c.equal_calibration(0.2, trained_tree, lambda row: row['race'] != race, lambda row: row['is_recid'] == '1', calib_pred, (7, 5))
c.conditional_statistical_parity(0.2, trained, lambda row: row['race'] != race, pos_pred, lambda x: (lambda row: int(row['priors_count']) > x), (0,))
c.predictive_equality(0.2, trained, lambda row: row['race'] != race, pos_pred, lambda row: row['is_recid'] == '1')
c.conditional_use_accuracy_equality(0.2, trained, lambda row: row['race'] != race, pos_pred, lambda row: row['is_recid'] == '1')
c.positive_balance(5, trained_tree, lambda row: row['race'] != race, sco_pred, lambda row: row['is_recid'] == '1')
c.negative_balance(5, trained_tree, lambda row: row['race'] != race, sco_pred, lambda row: row['is_recid'] == '1')
c.mean_difference(0.2, trained, lambda row: row['race'] != race, pos_pred)

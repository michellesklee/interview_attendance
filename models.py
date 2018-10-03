import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


df = pd.read_pickle('pickled.pkl')
df = pd.get_dummies(df)

# df.replace('no', False, inplace=True)
# df.replace('yes', True, inplace=True)
# df.replace('False.', False, inplace=True)
# df.replace('True.', True, inplace=True)
# df.drop(["State", "Area Code", "Phone"], axis=1, inplace=True)

y = df.pop('observed_attendance').values
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X,y)

# CV
def k_folds_CV(model, n_splits):
    scores = []
    kf = KFold(n_splits)
    for train_index, test_index in kf.split(X):
        Xk_train, Xk_test = X[train_index], X[test_index]
        yk_train, yk_test = y[train_index], y[test_index]
        model.fit(Xk_train, yk_train)
        preds = model.predict(Xk_test)
        score = precision_score(preds, yk_test)
        scores.append(score)
    cv_score = np.mean(scores)
    return cv_score

# Decision tree
dt = DecisionTreeClassifier(criterion="gini", splitter="best")
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

print('Decision Tree Accuracy: {}'.format(accuracy_score(dt_preds, y_test)))
print('Decision Tree Recall: {}'.format(recall_score(dt_preds, y_test)))
print('Decision Tree Precision: {}'.format(precision_score(dt_preds, y_test)))
cv = k_folds_CV(dt, n_splits=10)
print('CV DT Precision: {}'.format(cv))

# Random Forest
rf = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print('Random Forest Accuracy: {}'.format(accuracy_score(rf_preds, y_test)))
print('Random Forest Recall: {}'.format(recall_score(rf_preds, y_test)))
print('Random Forest Precision: {}'.format(precision_score(rf_preds, y_test)))
cv = k_folds_CV(rf, n_splits=10)
print('CV RF Precision: {}'.format(cv))


# Bagging
bag = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

bag.fit(X_train, y_train)
bag_preds = bag.predict(X_test)

print('Bagging Accuracy: {}'.format(accuracy_score(bag_preds, y_test)))
print('Bagging Recall: {}'.format(recall_score(bag_preds, y_test)))
print('Bagging Precision: {}'.format(precision_score(bag_preds, y_test)))
cv = k_folds_CV(bag, n_splits=10)
print('CV Bagging Precision: {}'.format(cv))

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=1)

knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

print('KNN Accuracy: {}'.format(accuracy_score(knn_preds, y_test)))
print('KNN Recall: {}'.format(recall_score(knn_preds, y_test)))
print('KNN Precision: {}'.format(precision_score(knn_preds, y_test)))
cv = k_folds_CV(knn, n_splits=10)
print('CV KNN Precision: {}'.format(cv))

# Gradient Boosting
boost = GradientBoostingClassifier(loss="deviance", learning_rate=0.1, n_estimators=100, subsample=1.0, criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort="auto")

boost.fit(X_train, y_train)
boost_preds = boost.predict(X_test)
print('Gradient Boosting Accuracy: {}'.format(accuracy_score(boost_preds, y_test)))
print('Gradient Boosting Recall: {}'.format(recall_score(boost_preds, y_test)))
print('Gradient Boosting Precision: {}'.format(precision_score(boost_preds, y_test)))
cv = k_folds_CV(boost, n_splits=10)
print('CV GB Precision: {}'.format(cv))

# Adaboost
ada = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm="SAMME.R", random_state=None)
ada.fit(X_train, y_train)
ada_preds = ada.predict(X_test)
print('AdaBoost Accuracy: {}'.format(accuracy_score(ada_preds, y_test)))
print('AdaBoost Recall: {}'.format(recall_score(ada_preds, y_test)))
print('AdaBoost Precision: {}'.format(precision_score(ada_preds, y_test)))
cv = k_folds_CV(ada, n_splits=10)
print('CV Ada Precision: {}'.format(cv))





# Decide Best model

# Grid search for best params

# ada example
gradient_boost_grid = {'n_estimators': [50, 100, 150, 200],
                      'random_state': [1, None],
                      'learning_rate': [0.1, .5, 1]}

gdr_gridsearch = GridSearchCV(AdaBoostClassifier(),
                             gradient_boost_grid,
                             n_jobs=-1,
                             verbose=True)
gdr_gridsearch.fit(X_train, y_train)

best_gdr_model = gdr_gridsearch.best_estimator_
best_gdr_model.fit(X_train, y_train)
best_gdr_preds = best_gdr_model.predict(X_test)

print("Best ADA Accuracy: {}".format(accuracy_score(best_gdr_preds, y_test)))
print("Best ADA Recall: {}".format(recall_score(best_gdr_preds, y_test)))
print("Best ADA Precision: {}".format(precision_score(best_gdr_preds, y_test)))
cv = k_folds_CV(best_gdr_model, n_splits=10)
print("CV Best ADA Precision: {}".format(cv))

# # Mega Super Ultra
msu = VotingClassifier(
    estimators= [('dt', dt), ('rf', rf), ('bag', bag), ('knn', knn), ('boost', boost), ('ada', ada), ('bgdr', best_gdr_model)], voting='soft')
msu.fit(X_train, y_train)
msu_preds = msu.predict(X_test)
print("Mega-Super-Ultra: {}".format(accuracy_score(msu_preds, y_test)))
# Important features (feature importance or partial dependence plot)

features = list(zip(np.asarray(df.columns), best_gdr_model.feature_importances_))


print(features)

importances = best_gdr_model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in boost.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# plot
 # yerr=std[indices],
plt.figure(figsize=(10,8))
plt.title("Features Importances")
plt.bar(range(X_test.shape[1]), importances[indices], color='r', align="center")
plt.xticks(range(X_test.shape[1]), np.asarray(df.columns)[indices], rotation=30)
plt.xlim([-1, X_test.shape[1]])
plt.savefig('features_importances.png')
plt.show()

"""
   A simple text classification model using ML classifiers
   1. Naive Mayes Classifier
   2. SVM based classifier
   A Grid Search was also performed for parameter tuning.
"""


from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# get the train and test dataset
all_data = fetch_20newsgroups(subset='all', shuffle=True)
x_train, x_test, y_train, y_test = train_test_split(all_data.data, all_data.target)
print('fetched the data ')
# Feature Extraction
tfidf_vectors = TfidfVectorizer()
x_train = tfidf_vectors.fit_transform(x_train)
x_test = tfidf_vectors.transform(x_test)


def prediction(classifier, test_data, test_labels):
    y_predict = classifier.predict(test_data)
    score = classifier.score(test_data, test_labels)

    return y_predict, score

###### Training Naive Bayes (NB) classifier on training data   ####

clf_nb = MultinomialNB()
clf_nb = clf_nb.fit(x_train, y_train)
predict_nb, score_nb = prediction(clf_nb, x_test, y_test)
print("Accuracy with Naive Bayes: ", score_nb*100)

##### Training Support Vector Machines - SVM #####

clf_svm = svm.LinearSVC()
clf_svm = clf_svm.fit(x_train, y_train)
predict_svm, score_svm = prediction(clf_svm, x_test, y_test)
print("Accuracy with SVM: ", score_svm*100)


##### Grid Search for parameter tuning #####
# defining parameter range
print(svm.LinearSVC().get_params().keys())
param_grid = [{'C': [1, 10], 'kernel': ['linear']}]
grid = GridSearchCV(svm.LinearSVC(), param_grid, refit=True, verbose=3, n_jobs=-1)
grid.fit(x_train, y_train)  # fitting the model for grid search

# # print best parameter after tuning
print(grid.best_params_)
grid_predictions = grid.predict(x_test)

print(classification_report(y_test, grid_predictions))

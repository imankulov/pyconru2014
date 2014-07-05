import numpy as np
import pandas as pd

imdb = pd.read_csv('data/movies3.csv', parse_dates=False, index_col=None).dropna()
top_genres = imdb.genre.value_counts().keys()[:5]
imdb = imdb.ix[imdb.genre.isin(top_genres)]
imdb.head()
np.random.seed(1234)
indexes = imdb.index.values.copy(); indexes[:10]
np.random.shuffle(indexes)
indexes[:10]
idx_train = indexes[:indexes.size / 2]; idx_test = indexes[indexes.size / 2: ]
print len(idx_train), len(idx_test)
train_db = imdb.ix[idx_train, :]
test_db = imdb.ix[idx_test, :]


from sklearn.metrics import mean_squared_error
mean_gross = train_db.worldwide_gross.mean(); mean_gross
real_gross = test_db.worldwide_gross.as_matrix(); real_gross[:10]
naive_estimated_gross = np.repeat(mean_gross, test_db.shape[0]); naive_estimated_gross[:4]
naive_error = mean_squared_error(real_gross, naive_estimated_gross); naive_error


from sklearn import linear_model
model = linear_model.LinearRegression()
X = train_db.ix[:, ['budget']].as_matrix(); X[:4]
Y = train_db.worldwide_gross.as_matrix(); Y[:4]
model.fit(X, Y)
predicted = model.predict(X); predicted[:4]
plot(X[:, 0], Y, 'k.', X[:, 0], predicted, 'r-')
print u'Если вы вложите в фильм $0, то вы получите %.2f миллионов $ дохода ;)' % model.intercept_
print u'В среднем каждый вложенный в картину доллар приносит %.2f долларов кассовых сборов' % model.coef_


Xtest = test_db.ix[:, ['budget']].as_matrix()
Ytest = train_db.worldwide_gross.as_matrix()
regression_estimated_gross = model.predict(Xtest)
plot(Xtest[:, 0], Ytest, 'k.', Xtest[:, 0], regression_estimated_gross, 'r-')
mean_squared_error(Ytest, regression_estimated_gross)
naive_error
train_dict = train_db.ix[:, ["genre", "rotten_tomatoes", "budget"]].to_dict('records'); train_dict[:2]
test_dict = test_db.ix[:, ["genre", "rotten_tomatoes", "budget"]].to_dict('records'); test_dict[:2]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
vec.fit(train_dict)
vec.get_feature_names()
train_arr = vec.transform(train_dict).toarray(); train_arr[:4]
test_arr = vec.transform(test_dict).toarray(); test_arr[:4]
model.fit(train_arr, Y)
model.intercept_; model.coef_
regression_estimated_gross2 = model.predict(test_arr)
print mean_squared_error(Ytest, regression_estimated_gross2), naive_error

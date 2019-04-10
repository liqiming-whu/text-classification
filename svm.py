import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from data_utils import *


word_dict = build_word_dict()
vocabulary_size = len(word_dict)
train_x, train_y = build_svm_dataset("train", word_dict)
test_x, test_y = build_svm_dataset("test", word_dict)

# print(type(train_x[1]))
# print(type(train_y[1]))


def addwhitespace(cont):
    c = []
    for i in cont:
        a = map(str, i)
        b = " ".join(a)
        c.append(b)
    return c


train_x = addwhitespace(train_x)
test_x = addwhitespace(test_x)

vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_x))
print('tfidf 矩阵:', tfidf.shape)

train_x = train_x[:10000]
train_y = train_y[:10000]
test_x = test_x[:5000]
test_y = test_y[:5000]

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(C=1, kernel='linear'))])
text_clf = text_clf.fit(train_x, train_y)
predicted = text_clf.predict(test_x)
print('准确率 ：', np.mean(predicted == test_y))

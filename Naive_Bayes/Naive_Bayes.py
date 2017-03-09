import os, sys
from collections import Counter
import numpy as np
import re
import random
from math import log

pos_path = "pos"
neg_path = "neg"
pos_dirs = os.listdir(pos_path)
neg_dirs = os.listdir(neg_path)

pos_reviews = []
neg_reviews = []
train = []

for review in pos_dirs:
    with open(os.path.join(pos_path, review),'r') as file:
        contents = file.read()
        pos_reviews.append([contents,1])

for review in neg_dirs:
    with open(os.path.join(neg_path, review), 'r') as file:
        contents = file.read()
        neg_reviews.append([contents, -1])

for review in pos_reviews:
    train.append(review)

for review in neg_reviews:
    train.append(review)

np.random.shuffle(train)
test = train[1000:]
train = train[0:1000]


def get_text(reviews, score):
    # Join together the text in the reviews for a particular tone.
    # We lowercase to avoid "Not" and "not" being seen as different words, for example.
    return " ".join([r[0].lower() for r in reviews if r[1] == score])

def count_text(text):
    # Split text into words based on whitespace.  Simple but effective.
    words = re.split("\s+", text)
    # Count up the occurence of each word.
    return Counter(words)

negative_text = get_text(train, -1)
positive_text = get_text(train, 1)

# Generate word counts for negative tone.
negative_counts = count_text(negative_text)

# Generate word counts for positive tone.
positive_counts = count_text(positive_text)


def get_y_count(score):
    # Compute the count of each classification occurring in the data.
    return len([r for r in train if r[1] == score])

# We need these counts to use for smoothing when computing the prediction.
positive_review_count = get_y_count(1)
negative_review_count = get_y_count(-1)

# These are the class probabilities (we saw them in the formula as P(y)).
prob_positive = float(positive_review_count) / float(len(train))
prob_negative = float(negative_review_count) / float(len(train))

def make_class_prediction(text, counts, class_prob, class_count):
    prediction = 0
    text_counts = Counter(re.split("\s+", text))
    for word in text_counts:
        # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
        # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
        # We also smooth the denominator counts to keep things even.
        prediction += log(float(text_counts.get(word)) * (float(counts.get(word, 0) + 1) / (float(sum(counts.values())) + float(class_count))))
        # Now we multiply by the probability of the class existing in the documents.

    return prediction + log(class_prob)

def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
      return -1
    return 1

# predictions = [make_decision(test[105][0], make_class_prediction)]
predictions = [make_decision(r[0], make_class_prediction) for r in test]
print("The predicted values are: {0}".format(predictions))

actual = [int(r[1]) for r in test]

print("The actual values are: {0}".format(actual))

from sklearn import metrics

# Generate the roc curve using scikits-learn.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))





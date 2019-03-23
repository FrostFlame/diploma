import pickle
import string

import pandas
import re

import csv
import lyricsgenius
import numpy
import textblob
import xgboost
from keras.preprocessing import text, sequence
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, ensemble
import scipy.sparse as sp


def main():
    genius = lyricsgenius.Genius("hu0VByrvYkRp8_UfZBOIl58Fho4_YnFY5RSs_Sh1tG2HwW9PY89P6MU13teEJtKV")
    artist = genius.search_artist("Andy Shauf", max_songs=1, sort="title")
    print([e.lyrics for e in artist.songs])


def classification():
    labels, texts = [], []
    with open('emotional_text_analysis/train_data.csv') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in reader:
            text = re.sub(r"@.+? ", '', row[1])
            texts.append(text)
            labels.append(row[0])
    texts = preprocess_text(texts)
    # le = preprocessing.LabelEncoder()
    # le.fit(labels)
    # with open('emotional_text_analysis/models/emotion_labels', 'xb') as file:
    #     pickle.dump(le, file=file)

    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])

    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect.transform(train_x)
    xvalid_count = count_vect.transform(valid_x)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                             max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
    xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

    trainDF['char_count'] = trainDF['text'].apply(len)
    trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
    trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
    trainDF['punctuation_count'] = trainDF['text'].apply(
        lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }

    # function to check and get the part of speech tag count of a words in a given sentence
    def check_pos_tag(x, flag):
        cnt = 0
        try:
            wiki = textblob.TextBlob(x)
            for tup in wiki.tags:
                ppo = list(tup)[1]
                if ppo in pos_family[flag]:
                    cnt += 1
        except:
            pass
        return cnt

    trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
    trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
    trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
    trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
    trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))

    def train_model(classifier, feature_vector_train, label, feature_vector_valid, model_name, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        pickle.dump(classifier, file=open('emotional_text_analysis/models/{}'.format(model_name), 'wb'))

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        return metrics.accuracy_score(predictions, valid_y)

    # Naive Bayes on Count Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count, 'NB, Count Vectors')
    print("NB, Count Vectors: ", accuracy)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, 'NB, WordLevel TF-IDF')
    print("NB, WordLevel TF-IDF: ", accuracy)

    # Naive Bayes on Ngram Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, 'NB, N-Gram Vectors')
    print("NB, N-Gram Vectors: ", accuracy)

    # Naive Bayes on Character Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, 'NB, CharLevel Vectors')
    print("NB, CharLevel Vectors: ", accuracy)

    # Linear Classifier on Count Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count, 'LR, Count Vectors')
    print("LR, Count Vectors: ", accuracy)

    # Linear Classifier on Word Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf, 'LR, WordLevel TF-IDF')
    print("LR, WordLevel TF-IDF: ", accuracy)

    # Linear Classifier on Ngram Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, 'LR, N-Gram Vectors')
    print("LR, N-Gram Vectors: ", accuracy)

    # Linear Classifier on Character Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y,
                           xvalid_tfidf_ngram_chars, 'LR, CharLevel Vectors')
    print("LR, CharLevel Vectors: ", accuracy)

    # SVM on Ngram Level TF IDF Vectors
    accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, 'SVM, N-Gram Vectors')
    print("SVM, N-Gram Vectors: ", accuracy)

    # RF on Count Vectors
    accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count, 'RF, Count Vectors')
    print("RF, Count Vectors: ", accuracy)

    # RF on Word Level TF IDF Vectors
    accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf, 'RF, WordLevel TF-IDF')
    print("RF, WordLevel TF-IDF: ", accuracy)

    # Extereme Gradient Boosting on Count Vectors
    accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc(), 'Xgb, Count Vectors')
    print("Xgb, Count Vectors: ", accuracy)

    # Extereme Gradient Boosting on Word Level TF IDF Vectors
    accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc(), 'Xgb, WordLevel TF-IDF')
    print("Xgb, WordLevel TF-IDF: ", accuracy)

    # Extereme Gradient Boosting on Character Level TF IDF Vectors
    accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y,
                           xvalid_tfidf_ngram_chars.tocsc(), 'Xgb, CharLevel Vectors')
    print("Xgb, CharLevel Vectors: ", accuracy)

    features = sp.hstack((xtrain_count, xtrain_tfidf, xtrain_tfidf_ngram, xtrain_tfidf_ngram_chars), format='csr')
    valids = sp.hstack((xvalid_count, xvalid_tfidf, xvalid_tfidf_ngram, xvalid_tfidf_ngram_chars), format='csr')

    accuracy = train_model(naive_bayes.MultinomialNB(), features, train_y, valids, 'NB, All features')
    print("NB, All features: ", accuracy)

    # Linear Classifier on Count Vectors
    accuracy = train_model(linear_model.LogisticRegression(), features, train_y, valids, 'LR, All features')
    print("LR, All features: ", accuracy)

    # SVM on Ngram Level TF IDF Vectors
    accuracy = train_model(svm.SVC(), features, train_y, valids, 'SVM, All features')
    print("SVM, All features: ", accuracy)

    # RF on Count Vectors
    accuracy = train_model(ensemble.RandomForestClassifier(), features, train_y, valids, 'RF, All features')
    print("RF, All features: ", accuracy)

    # Extereme Gradient Boosting on Count Vectors
    accuracy = train_model(xgboost.XGBClassifier(), features.tocsc(), train_y, valids.tocsc(), 'Xgb, All features')
    print("Xgb, All features: ", accuracy)


def preprocess_text(data):
    result = []
    for text in data:
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in tokens]
        result.append(' '.join(stemmed))
    return result


def predict(classifier, data):
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                             max_features=5000)
    texts = []
    with open('train_data.csv') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in reader:
            texts.append(re.sub(r"@.+? ", '', row[1]))
    tfidf_vect_ngram_chars.fit(texts)

    data = tfidf_vect_ngram_chars.transform(preprocess_text(data))
    s = classifier.predict(data)
    le = pickle.load(open('models/emotion_labels', 'rb'))
    # print(list(le.inverse_transform(s)))
    return list(le.inverse_transform(s)), le.classes_


if __name__ == '__main__':
    # classification()
    model = pickle.load(open('models/Xgb, CharLevel Vectors', 'rb'))
    s = "I am not the only traveler " \
        "Who has not repaid his debt " \
        "I've been searching for a trail to follow again " \
        "Take me back to the night we met " \
        "And then I can tell myself " \
        "What the hell I'm supposed to do " \
        "And then I can tell myself " \
        "Not to ride along with you " \
        "I had all and then most of you " \
        "Some and now none of you " \
        "Take me back to the night we met " \
        "I don't know what I'm supposed to do " \
        "Haunted by the ghost of you " \
        "Oh, take me back to the night we met " \
        "When the night was full of terrors " \
        "And your eyes were filled with tears " \
        "When you had not touched me yet " \
        "Oh, take me back to the night we met " \
        "I had all and then most of you " \
        "Some and now none of you " \
        "Take me back to the night we met " \
        "I don't know what I'm supposed to do " \
        "Haunted by the ghost of you " \
        "Take me back to the night we met"
    predict(model, ['what is wrong', 'i am angry', 'happy', s, 'i hate you'])

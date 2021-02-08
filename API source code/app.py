import pickle
import re

import joblib
import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)


def preprocess(tweet):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    stopword_list = nltk.corpus.stopwords.words('english') + ['u', 'im', 'rt', 'ummm', 'b', 'dont', 'arent', 'ya',
                                                              'yall', 'isnt',
                                                              'cant', 'couldnt', 'wouldnt', 'wont', 'yr', 'aint',
                                                              'gonna', 'ur',
                                                              'didnt', 'r', 'wasnt', 'werent', 'might', 'maybe',
                                                              'doesnt', 'would', 'shes', 'hes', 'youre', 'omg', 'us',
                                                              'wow'] + stopwords.words('english')

    preposition = ['in', 'at', 'by', 'from', 'on', 'for', 'with', 'about', 'into', 'through', 'between', 'under',
                   'against', 'during', 'without', 'upon', 'toward', 'among', 'within', 'along', 'across', 'behind',
                   'near', 'beyond', 'using', 'throughout', 'despite', 'to', 'beside', 'plus', 'towards', 'concerning',
                   'onto', 'beneath', 'via']
    stopword_list += preposition
    ps = PorterStemmer()

    tweet = tweet.lower()
    # remove unwanted characters
    tweet = re.sub(r'(\\x[^\s][^\s])', "", tweet)
    # remove \n
    tweet = re.sub(r'\\n', ' ', tweet)
    # remove url and mentions
    tweet = re.sub(r"(?:\@|rt @|https?\://)\S+", " ", tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # tokenize tweets and remove punctuations
    tokens = tokenizer.tokenize(tweet)
    # remove stopwords and stemming
    tokens = [ps.stem(word) for word in tokens if word not in stopword_list]
    # remove short words and numbers
    twt = ' '.join([token for token in tokens if token.isalpha() and len(token) > 2])
    return twt


def hashtag_recommend(tweet, hashtag_frequency, inverse_corpus_frequency):
    tweet_score = []
    for word in tweet.split():
        score = 0
        if word in hashtag_frequency:
            for hashtag in hashtag_frequency[word]:
                if hashtag in inverse_corpus_frequency:
                    score += hashtag_frequency[word][hashtag] * inverse_corpus_frequency[hashtag]
                else:
                    score += hashtag_frequency[word][hashtag]
        tweet_score.append((word, score))

    tweet_score = sorted(tweet_score, key=lambda tweet_score: tweet_score[1], reverse=True)
    return tweet_score


def hf_ihu_recommendation(new_tweet_str):
    inverse_corpus_frequency = pickle.load(open('inverseCorpusFrequency.pickle', 'rb'))
    hashtag_frequency = pickle.load(open('hashtagFrequency.pickle', 'rb'))
    sorted_word_hashtags = pickle.load(open('sortedWordHashtags.pickle', 'rb'))

    ranked_words = hashtag_recommend(new_tweet_str, hashtag_frequency, inverse_corpus_frequency)
    hashtag_recommended = []
    num = 0
    less = 0
    for i in range(0, len(ranked_words)):
        num_per_word = 0
        if num == 5 or len(hashtag_recommended) >= 5:
            break
        hashtag_num = 0
        if i < 4:
            hashtag_num = 4 - i + less
        else:
            hashtag_num = 2
        if ranked_words[i][1] != 0:
            for j in range(0, hashtag_num):
                if len(hashtag_recommended) >= 5:
                    break
                if j == len(sorted_word_hashtags[ranked_words[i][0]]):
                    break
                hashtag_recommended.append(sorted_word_hashtags[ranked_words[i][0]][j][0])
                num += 1
                num_per_word += 1
            less = hashtag_num - num_per_word

        else:
            break
    return hashtag_recommended


def find_features(document, word_features):
    words = set(document)
    feature_set = {}
    for w in word_features:
        feature_set[w] = (w in words)
    return feature_set


@app.route('/predict/', methods=['POST'])
def predict():
    features_file = open('word_features_file_2k.pickle', 'rb')
    features = pickle.load(features_file)

    model_file = open('multinumialNB_classifier_2k.pickle', 'rb')
    model = joblib.load(model_file)

    data = request.json["message"]
    processed = preprocess(data)
    nb_prediction = model.classify(find_features(processed.split(), features))
    hf_prediction = hf_ihu_recommendation(processed)
    
    
    lst = [nb_prediction] + hf_prediction
    result = []
    for hash in lst:
        if hash not in result:
            result.append(hash)
    
    

    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

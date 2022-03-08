from pulsar import Function
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('subjectivity')
nltk.download('punkt')
nltk.download('vader_lexicon')

class SentimentFunction(Function):
  def __init__(self):
    pass

  def process(self, input, context):

    originalText = input

    n_instances = 100
    subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
    # len(subj_docs), len(obj_docs)

    train_subj_docs = subj_docs[:80]
    test_subj_docs = subj_docs[80:100]
    train_obj_docs = obj_docs[:80]
    test_obj_docs = obj_docs[80:100]
    training_docs = train_subj_docs+train_obj_docs
    testing_docs = test_subj_docs+test_obj_docs
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
    
    # We use simple unigram word features, handling negation:

    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    
    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)
    
    # We can now train our classifier on the training set, and subsequently output the evaluation results:

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)


    sentences = []
    paragraph = str(originalText)
    from nltk import tokenize
    lines_list = tokenize.sent_tokenize(paragraph)
    sentences.extend(lines_list)
    sid = SentimentIntensityAnalyzer()
    result = ""
    sentences_dictionary = {"sentences":[]}
    res = sentences_dictionary["sentences"]

    for sentence in sentences:
        # print(sentence)
        ss = sid.polarity_scores(sentence)
        compound = ""
        negative = ""
        neutral = ""
        positive = ""
        for k in sorted(ss):
            result = result + '{0}: {1} '.format(k, ss[k])
            
            if k == "compound":
                compound = str(ss[k])
            if k == "neg":
                negative = str(ss[k])
            if k == "neu":
                neutral = str(ss[k])
            if k == "pos":
                positive = str(ss[k])
            
        x = {
            "sentence": sentence,
            "compound": compound,
            "positive": positive,
            "negative": negative,
            "neutral": neutral
        }

        res.append(x)

    return str(res)
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
    # print (len(unigram_feats))
    # 83
    # sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    # We apply features to obtain a feature-value representation of our datasets:

    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)
    # We can now train our classifier on the training set, and subsequently output the evaluation results:

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)


    sentences = []
    # paragraph = "A few years ago, we identified a serious communications gap between corporates and investors over environmental, social and governance (ESG) information. Investors were looking for standardized, rigorous data to support investment decisions. Many corporates, however, were releasing ESG information inconsistently and in a manner investors found difficult to use. Since then, this gap has continued and ESG’s importance has grown. More and more institutional investors are looking for a company’s management to articulate a sustainable long-term value creation strategy that outlines not just growth opportunities, but also the related risks. They view ESG matters as critical to understanding the full risk profile of a company and how prepared it is for the future. There’s good reason for investors to put this emphasis on ESG questions. Companies with risk management practices that take into consideration broader industry, regulatory and societal risks are more likely to drive long-term sustainable performance—and shareholder value."
    # paragraph = "A very bad idea! This is happy place with some dummy people"
    paragraph = str(originalText)
    from nltk import tokenize
    lines_list = tokenize.sent_tokenize(paragraph)
    sentences.extend(lines_list)
    # tricky_sentences = []
    # sentences.extend(tricky_sentences)
    sid = SentimentIntensityAnalyzer()
    result = ""
    # resultsJsonData = '{}'
    # resultsJson = json.loads(resultsJsonData)
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
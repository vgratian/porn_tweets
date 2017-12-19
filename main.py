import string, timeit
from math import log10, sqrt

"""
This program compares the input data (a collection of archivized tweets) to a
benchmark (a selection of 3 tweets) and sorts them by cosine similiarity. The
the top 100 results are printed.

By default, we choose tweets with adult content and expect that the top 100
results will have a high probability of adult content as well. The benchmark
tweets can be changed to do the same experiment with other genres of tweets.

The default 3 benchmark tweets are displayed in the dictionary below
(tweets are normalized and URL's have been moved):
"""

global benchmark, fp_tweets, fp_stopwords, stopwords, collection_size
global terms, terms_tf, terms_df, terms_idf, terms_tf_idf, matched_tweets, top

benchmark = {
    '766419051145035777': 'First anal sex in the kitchen #amateur #anal #blond #blowjob #brunette #cumshots',
    '776913092257148928': '#women sleeping sex porn actor nude',
    '779008212288823296': '#dark pussy hair teens lesbian movies' }
collection_size = 0 # total number of tweets
stopwords = []
terms = [] # dict with terms as keys and idf as values
terms_tf = []
terms_df = []
terms_idf = []
terms_tf_idf = []
matched_tweets = {} # final result, format: {'tweet' : 'cos_sim score'}
top = [] # will contain top 100 tweets by decreasing cos_sim score
fp_tweets = 'data/tweets'
fp_stopwords = 'data/stopwords'

def run():
    """
    Main routine.
    """
    # start counting runtime
    start = timeit.default_timer()

    # populate global variables
    import_stopwords()
    merge_benchmark()

    # compute tf scores for terms against benchmark
    global terms_tf
    terms_tf = [1/len(terms) for term in terms]

    # compute idf scores for terms
    get_idf_scores()

    # print terms and df-s
    print('{} terms in benchmark. Printing terms with document frequency (df):'
        .format(len(terms)))
    for term, df in zip(terms, terms_df):
        print('({}) {}'.format(df, term))
    print('Ranking {} tweets by cosine similarity.'.format(collection_size))

    # compute tf-idf scores for terms
    global terms_tf_idf
    terms_tf_idf = [tf*idf for tf, idf in zip(terms_tf, terms_idf)]

    # now compute tf-idf and cosine similiarity for tweets
    get_tf_idf_scores()

    # sort matched tweets and slice top 100 results
    top = sorted(matched_tweets.items(), key=lambda x:x[1], reverse=True)[:100]

    # end runtime counting
    stop = timeit.default_timer()
    runtime = stop - start

    # print runtime and top 100 results
    print('Found {} matching tweets in {} seconds.'.format
        (len(matched_tweets), round(runtime, 2)))
    print('Printing 100 tweets with highest cosine similarity...')
    for tweet in top:
        print('({})\t{}'.format(tweet[1], tweet[0]))


def get_tf_idf_scores():
    global terms, terms_idf, matched_tweets
    tweets = read_tweets()
    for tweet in tweets:
        if len(tweet.split('\t')) == 5: # Skip tweets in wrong format or missing data
            tweet_text = tweet.split('\t')[4]
            tweet_tokens = tokenize_tweet(tweet_text)
            tweet_id = tweet.split('\t')[1]
            if tweet_tokens and tweet_id not in benchmark.keys(): # Skip tweets with no tokens
                tf_idf_list = []
                at_least_one_match = False
                for term in terms:
                    tf = 0
                    for token in tweet_tokens:
                        if term == token:
                            tf += 1
                            at_least_one_match = True
                    tf = tf/len(tweet_tokens)
                    tf_idf = tf * terms_idf[terms.index(term)]
                    tf_idf_list.append(tf_idf)
                if at_least_one_match:
                    cos_sim = get_cos_sim(tf_idf_list)
                    matched_tweets[tweet_text] = cos_sim


def get_cos_sim(tweet_vectors):
    dot_prod = get_dot_prod(tweet_vectors)
    eucl_len_prod = get_eucl_len_prod(tweet_vectors)

    return dot_prod / eucl_len_prod


def get_dot_prod(tweet_vectors):
    global terms_tf_idf
    dot_prod = 0
    for a, b in zip(terms_tf_idf, tweet_vectors):
        dot_prod += a * b
    return dot_prod


def get_eucl_len_prod(tweet_vectors):
    global terms_tf_idf
    eucl_len0 = 0
    eucl_len1 = 0
    for a, b in zip(terms_tf_idf, tweet_vectors):
        eucl_len0 += pow(a, 2)
        eucl_len1 += pow(b, 2)
    return sqrt(eucl_len0) * sqrt(eucl_len1)


def get_idf_scores():
    """
    Computes document frequency for terms (i.e. number of tweets containing the
    term). Also sets the collection size. Both variables are globally changed.
    """
    global terms, terms_idf, terms_df, collection_size
    tweets = read_tweets()

    # First get document frequency (df)
    terms_df = [0 for each_term in terms]
    collection_size = 0
    for tweet in tweets:
        if len(tweet.split('\t')) == 5:
            tweet_tokens = tokenize_tweet(tweet.split('\t')[4])
            tweet_id = tweet.split('\t')[1]
            if tweet_id not in benchmark.keys():
                collection_size += 1
                for term in terms:
                    if term in tweet_tokens:
                        terms_df[terms.index(term)] += 1
    # Get idf from df
    for df in terms_df:
        idf = 1 + log10( collection_size / df ) if df != 0 else 0
        terms_idf.append(idf)


def merge_benchmark():
    global terms, benchmark
    for tweet in benchmark:
        tokens = tokenize_tweet(benchmark[tweet]) # returns list of tokens
        for word in tokens:
            if word not in terms:
                terms.append(word)


def import_stopwords():
    """
    Import stop words from file into global variable stopwords
    """
    global fp_stopwords, stopwords
    try:
        f = open(fp_stopwords, 'r')
    except:
        print('Can\'t read stopwords file')
        raise IOError
    else:
        stopwords = f.read().splitlines()
    finally:
        f.close()


def tokenize_tweet(tweet):
    """
    Minimal tokenization interface. Splits string into words by space. Removes
    punctuation, URLs, removes stop words. Returns only alphabetical string
    words, lowercased.
    """
    global stopwords
    removepunct = str.maketrans('', '', string.punctuation)
    dirty_tokens = tweet.split()
    clean_tokens = []
    for token in dirty_tokens:
        token = token.translate(removepunct).lower()
        if token.isalpha() and token not in stopwords and 'http' not in token:
            clean_tokens.append(token)
    return clean_tokens


def read_tweets():
    """
    Generator that returns tweets from file one by one
    """
    global fp_tweets
    try:
        f = open(fp_tweets, 'r')
    except:
        print('Can\'t read tweet file')
        raise IOError
    else:
        for line in f.readlines():
            yield line
    finally:
        f.close()


if __name__ == '__main__':
    run()

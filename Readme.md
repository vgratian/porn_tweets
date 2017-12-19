
# porn_tweets

Simple Python program to calculate cosine similarity between tweets. The program takes as input an file with (preferably large number of) tweets.
We used an archive with 1M tweets from 2016. A small subset of this collection is separated as `benchmark`, the rest of the collection is then compared to the benchmark and ranked by decreasing cosine similarity.

This code is a homework assignment for the course Information Retrieval and Text Mining at the University of Stuttgart. I thank my classmate Jyothsna Vijayendra for explaining the computation of cosine similarity.

## benchmark & results

We chose for our assignment 3 tweets with adult content as benchmark. We were curious is the tweets with high cosine similarity to this benchmark would also be of adult content. The resulting top 100 tweets all belonged to the pornographic genre.

The ranking of 1M tweets required 130.22 seconds from a XPS Dell laptop with i5 processor and 8GB memory.

## stopwords
The list of 119 stop words copied and adapted from [Stanford Core NLP library](https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt)
which is under the GNU General Public license.

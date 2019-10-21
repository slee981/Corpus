import pandas as pd 
import sys 

sys.path.append('../')
import Corpus as cp

docs = [
    'This is a long and tedius document', 
    'This is also a document 432d.', 
    'A pig flew into my house to eat dinner', 
    'I roasted and ate a pig for dinner'
]

d = pd.DataFrame()
d['text'] = docs 
d = d.set_index(pd.Index([1, 2, 3, 4]))

corpus = cp.Corpus(d['text'])
corpus.wordcloud()
corpus.wordcloud(ngram=1)
corpus.fit(ngram=1)
corpus.wordcloud()
corpus.show_lda_topics()
corpus.show_lda_topics(n_words=10)
corpus.show_lda_topics(ngram=2)
corpus.show_lda_topics(n_topics=3)
corpus.show_lda_topics(ngram=1, n_topics=4)
corpus.fit_lda(ngram=1, n_topics=10)
i = corpus.get_index('pig')
i 
corpus.get_ngram_ct('pig')
corpus.get_ngram_ct('pig', 1)

corpus.show_top_ngrams()
corpus.show_top_ngrams(n=5)
corpus.show_lda_topics()
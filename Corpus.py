#######################################################################
# Author : Stephen Lee 
# Date   : 9.23.19
#######################################################################

import pandas as pd 
import matplotlib.pyplot as plt
from gensim.models import LdaMulticore as LDA
from gensim.models import TfidfModel as TFIDF
from gensim.corpora import Dictionary 
from gensim.parsing.porter import PorterStemmer
from wordcloud import WordCloud

from collections import defaultdict
from datetime import datetime
import os, sys, csv

class Corpus:

    #################################################################################
    # Storage
    
    df = pd.DataFrame()
    ps = PorterStemmer()

    #################################################################################
    # Constructor

    def __init__(self, text_df, ngram=2, stopwords=True, stem=True):
        self.df['raw_text'] = text_df
        self.df['stemmed'] = self.df['raw_text'].apply(self._process_raw_text, args=[stopwords, stem])
        self.fit(ngram=ngram)
        try: 
            self.df = self.df.set_index(text_df.index)
        except: 
            pass


    #################################################################################
    # External Functions

    # setters

    def fit(self, ngram, vocab_size=10000): 

        self.ngram = ngram

        # reset values 
        self.ngram_to_counts = defaultdict(int)
        self.ngram_to_index = defaultdict(int)
        self.vocabulary = [] 

        # fit to make ngrams 
        self.df['ngrams'] = self.df['stemmed'].apply(self._make_ngrams, args=[ngram])

        # count all ngrams for document, make vocab
        self.df['ngrams'].apply(self._count_all)          # fills self.ngram_to_counts
        self.vocabulary = self._make_vocab(vocab_size)    # fills vocabulary 

        # make bag of ngrams
        self.df['bag_ngrams'] = self.df['ngrams'].apply(self._bag_of_ngrams)

        self.fit_lda()


    def fit_lda(self, n_topics=10, alpha=5, eta=0.025, ngram=None):
        if isinstance(ngram, int) and ngram != self.ngram:
            self.fit(ngram)
        self.num_lda_topics = n_topics
        self._fit_gensim_params()
        self.lda = LDA(corpus=self.gensim_corpus, id2word=self.gensim_dict, num_topics=n_topics, 
                        passes=2, workers=2, alpha=alpha, eta=eta)

    
    def wordcloud(self, ngram=None):
        if not isinstance(ngram, int): 
            ngram = self.ngram 

        self._raw_txt_to_ngrams(ngram)
        wdcloud = WordCloud(background_color='white',
                            max_font_size=56,
                            color_func=self._wc_color_func
                  ).generate_from_frequencies(self.raw_ngram_to_counts)
        plt.imshow(wdcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        return wdcloud
    
    
    def save_wordcloud(self, ngram=2):
        wdcloud = self.wordcloud(ngram)

        # ask to save 
        ans = input("Would you like to save the wordcloud? ('y' or 'n')\n  $ ")
        if ans.lower() == 'y':
            fname = input("Please enter the filename to save as (e.g. 'wordcloud-1-gram.png')\n  $ ")
            try:
                wdcloud.to_file(fname)
            except: 
                print('Cannot save under that file name. Make sure the path exists.')
                print("Saving to working directory as 'wordcloud.png'")
                wdcloud.to_file('wordcloud.png')


    # getters

    def get_index(self, ngram): 
        return self.ngram_to_index.get(ngram, 'ngram not in vocabulary.')


    def get_bag(self, doc_idx): 
        return self.df['bag_ngrams'].loc[doc_idx]

    
    def get_ngram_ct(self, ngram_str, doc_idx=None):
        if isinstance(doc_idx, int):     
            ngram_idx = self.get_index(ngram_str)
            doc_bag = self.get_bag(doc_idx)
            return doc_bag[ngram_idx]
        else: 
            return self.ngram_to_counts.get(ngram_str, 'ngram not in vocabulary')


    def get_top_ngrams(self, n=10, ngram=None): 
        if isinstance(ngram, int) and ngram != self.ngram:
            self.fit(ngram)
        
        top_ngrams = {}
        for ngram, ct in sorted(self.ngram_to_counts.items(), key=lambda x: x[1], reverse=True)[:n]: 
            top_ngrams[ngram] = ct 
        return top_ngrams


    def get_lda_topics(self, n_words=5, ngram=None, n_topics=None):

        # refit the corpus if ngram specified is differen that current fit
        if isinstance(ngram, int) and ngram != self.ngram:
            self.fit(ngram)

        # refit the lda if n_topcis specified is different than current fit
        if isinstance(n_topics, int) and n_topics != self.num_lda_topics:
            self.fit_lda(n_topics=n_topics)

        # display
        return self.lda.show_topics(num_words=n_words)

    
    def show_top_ngrams(self, n=10, ngram=None):
        top_ngrams = self.get_top_ngrams(n, ngram)
        print('\n############ TOP NGRAMS ############')
        for word, ct in top_ngrams.items(): 
            print('{}: {}'.format(word, ct))
        print('\n\n')


    def show_lda_topics(self, n_words=5, ngram=None, n_topics=None):
        topics = self.get_lda_topics(n_words, ngram, n_topics)
        print('\n############ LDA TOPICS ############')
        for tnum, twords in topics: 
            print('{}: {}'.format(tnum, twords))
        print('\n\n')


    def export_topics(self, fname, n_words=10, ngram=None, n_topics=None): 
        topics = self.get_lda_topics(n_words, ngram, n_topics)
        headers = self._make_topic_headers(n_words)
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for tnum, twords in topics: 
                words = [w for w in twords.split('"') if '*' not in w and w != '']
                words.insert(0, tnum)         # add topic number to start of list 
                writer.writerow(words)


    def export_top_ngrams(self, fname, n=10, ngram=None): 
        top_ngrams = self.get_top_ngrams(n, ngram)
        headers = ['Word', 'Count']
        with open(fname, 'w', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow(headers)
            for word, ct in top_ngrams.items(): 
                writer.writerow([word, ct])


    #################################################################################
    # Internal Functions

    # process text

    def _process_raw_text(self, txt, stopwords, stem): 
        if isinstance(txt, str):
            t = self._only_alphas(txt)
            if stopwords: 
                t = self._remove_stop_words(t)
            if stem: 
                t = self._stem(t)
        else: 
            t = ''
        return t


    def _only_alphas(self, txt):
        if isinstance(txt, str):
            txt = str(txt)
            txt = txt.replace('\n', ' ')
            t = [i.lower() for i in txt if i.isalpha() or i == ' ']
        else: 
            t = ''
        return ''.join(t)


    def _remove_stop_words(self, txt):
        stop_words = self._get_stop_words() 
        txt = txt.split(' ')
        return ' '.join([w.strip() for w in txt if ((w not in stop_words) and len(w) > 2) ])


    def _get_stop_words(self):
        fpath = os.path.join(os.path.dirname(__file__), '.stopwords/stopwords.txt')
        with open(fpath, 'r') as f: 
            txt = f.read()
        return txt.split('\n')


    def _stem(self, txt): 
        txt = txt.split(' ')
        return ' '.join([self.ps.stem(w) for w in txt])

    # count ngrams, create vocabulary, create bag of ngrams

    def _make_ngrams(self, txt, ngram):
        '''
            Input   ->  txt, one document
                    ->  ngram, the number of 'grams' per token 
            Output  ->  ngram representation of the document 
                        e.g.: 
                        "the fox ran wild" => "the.fox,fox.ran,ran.wild"    
        '''
        token = [t for t in txt.split(' ')]
        ngrams = zip(*[token[i:] for i in range(ngram)])
        ngrams_lst = ['_'.join(g) for g in ngrams]
        return ','.join(ngrams_lst)


    def _count_all(self, txt):
        if isinstance(txt, str):
            ngrams = txt.split(',')
        else: 
            ngrams = ['']

        for ngram in ngrams:
            if len(ngram) > 1:
                self.ngram_to_counts[ngram] += 1
        return


    def _count_all_raw(self, txt):
        if isinstance(txt, str):
            ngrams = txt.split(',')
        else: 
            ngrams = ['']

        for ngram in ngrams:
            if len(ngram) > 1:
                self.raw_ngram_to_counts[ngram] += 1
        return


    def _make_vocab(self, vocab_size): 
        vocab = []
        total_words = 0
        for ngram, _ in sorted(self.ngram_to_counts.items(), key=lambda x: x[1], reverse=True): 
            if total_words < vocab_size and len(ngram) > 1:
                vocab.append(ngram)
                self.ngram_to_index[ngram] = total_words 
                total_words += 1
            else: 
                break 
        return vocab


    def _bag_of_ngrams(self, ngram_doc):
        bag = [0] * len(self.vocabulary)   # empty bag

        # check type, this happens if doc = NaN in pandas
        if isinstance(ngram_doc, float):
            return bag
        elif isinstance(ngram_doc, str):
            doc_cts = defaultdict(int)
            doc = ngram_doc.split(',')
            for ngram in doc: 
                doc_cts[ngram] += 1        # count ngrams in doc
            for ngram in doc_cts:
                vocab_index = self.ngram_to_index.get(ngram, -1)
                if vocab_index != -1: 
                    bag[vocab_index] = doc_cts[ngram]
        return bag

    
    # wordcloud helper 
    def _wc_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        # hard coded as max font size = 56
        r = int(max(min(862 / font_size - 15, 255), 0))    # magic, 0 for big font, 255 for small font
        g = r // 10
        b = 255 - g
        return "rgb({}, {}, {})".format(r, g, b)

    
    def _raw_txt_to_ngrams(self, ngram, vocab_size=10000, remove_stopwords=True): 

        # reset values 
        self.raw_ngram_to_counts = defaultdict(int)
       
        # fit to make ngrams 
        tmp_txt_df = self.df['raw_text'].apply(self._only_alphas)
    
        if remove_stopwords: 
            tmp_txt_df = tmp_txt_df.apply(self._remove_stop_words)

        # count all ngrams for document, make vocab
        tmp_ngram_df = tmp_txt_df.apply(self._make_ngrams, args=[ngram])
        tmp_ngram_df.apply(self._count_all_raw)   # fills self.raw_ngram_to_counts


    # lda helpers 
    def _ngram_str_to_lst(self, ngram_str):
        return [w for w in ngram_str.split(',') if len(w) > 2]


    def _make_topic_headers(self, n_words): 
        headers = ['Topic_Num']
        for i in range(n_words): 
            headers.append('Word_{}'.format(i + 1))
        return headers


    # gensim helpers 
    def _fit_gensim_params(self):
        doc_ngrams = self.df['ngrams'].apply(self._ngram_str_to_lst)
        self.gensim_dict = Dictionary(doc_ngrams)
        self.gensim_corpus = [self.gensim_dict.doc2bow(doc) for doc in doc_ngrams]
        self.tfidf = TFIDF(corpus=self.gensim_corpus, id2word=self.gensim_dict)

        
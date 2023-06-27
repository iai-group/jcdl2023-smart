import re

import gensim.downloader
from nltk.corpus import stopwords
from scipy import spatial

from .features import Features

WORD_EMBEDDINGS = 'word2vec-google-news-300'

STOPWORDS = stopwords.words('english')

# Mapping from British to American English
UK_TO_US = {
    'theatre': 'theater',
    'organisation': 'organization',
    'colour': 'color',
    'flavour': 'flavor',
    'honours': 'honors'
}


class W2VFeatures(Features):

    def __init__(self, types):
        super().__init__('QT', types)
        print('Loading word embeddings {}... '.format(WORD_EMBEDDINGS), end='')
        self.__w2v = gensim.downloader.load(WORD_EMBEDDINGS)
        print('done')

    def __get_type_terms_w2v(self, type_id):
        """Parses an input (CamelCase) type label and returns a list of terms
         for which w2v embeddings exist. Stopwords and OOV terms are filtered
         out."""
        type_terms = []
        for term in re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))',
                               type_id[4:]):
            # Keep acronyms unchanged, otherwise lowercase
            if not term.isupper():
                term = term.lower()
                if term in STOPWORDS:
                    continue
            if term in self.__w2v:
                type_terms.append(term)
            elif term in UK_TO_US:
                if UK_TO_US[term] in self.__w2v:
                    type_terms.append(UK_TO_US[term])
            else:
                # print('Term not found: {}'.format(term))
                pass
        return type_terms

    def __get_query_terms_w2v(self, query):
        """Parses query and return a list of terms for which w2v embeddings
        exist. Tries different surface variations of terms."""
        # Strip punctuation
        for p in ['.', ',', '?', '!', ':', ';', '"', '\'s', '\'', '\n']:
            query = query.replace(p, '')

        query_terms = []
        for term in query.split():
            if term.lower() in STOPWORDS:
                continue
            # Allcaps
            if term.isupper() and term in self.__w2v:
                query_terms.append(term)
            # Try lower-cased
            elif term.lower() in self.__w2v:
                query_terms.append(term.lower())
            # Try original
            elif term in self.__w2v:
                query_terms.append(term)
            else:
                term = term.lower()
                # Try US spelling
                if term in UK_TO_US:
                    if UK_TO_US[term] in self.__w2v:
                        query_terms.append(UK_TO_US[term])
                # Try capitalized
                if term.capitalize() in self.__w2v:
                    query_terms.append(term.capitalize())
                else:
                    # print('Term not found: {}'.format(term))
                    pass
        return query_terms

    def __get_centroid_vector(self, terms):
        sum_vec = [0] * len(self.__w2v[terms[0]])
        for term in terms:
            for i, w in enumerate(self.__w2v[term]):
                sum_vec[i] += w
        return [s / len(terms) for s in sum_vec]

    @staticmethod
    def cosine(v1, v2):
        """Returns the cosine similarity between the two vectors."""
        return 1 - spatial.distance.cosine(v1, v2)

    def get_features(self, query, type_id):
        # Parse query and type labels; resulting lists contain only terms for
        # which embeddings exist
        query_terms = self.__get_query_terms_w2v(query)
        type_terms = self.__get_type_terms_w2v(type_id)

        # By definition, we don't compete the w2v features term vectors cannot
        # be found for none of the type or query terms
        if len(type_terms) == 0 or len(query_terms) == 0:
            return {
                'w2v_aggr_cos_sim': None,
                'w2v_max_cos_sim': None,
                'w2v_avg_cos_sim': None
            }

        type_centroid = self.__get_centroid_vector(type_terms)
        query_centroid = self.__get_centroid_vector(query_terms)
        termwise_sims = []
        for type_term in type_terms:
            for query_term in query_terms:
                termwise_sims.append(
                    self.cosine(self.__w2v[type_term], self.__w2v[query_term]))

        return {
            'w2v_aggr_cos_sim': self.cosine(type_centroid, query_centroid),
            'w2v_max_cos_sim': max(termwise_sims),
            'w2v_avg_cos_sim': sum(termwise_sims) / len(termwise_sims)
        }

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.corpus import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download("stopwords")
nltk.download("wordnet")


class Posts:
    def __init__(self, text_df):
        self.text_df = text_df
        return

    def tokenization(self):
        tqdm.pandas()
        tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}')
        tokens_df = self.text_df.progress_apply(
            lambda x: tokenizer.tokenize(x.lower()))
        return tokens_df

    def remove_stopwords(self, tokens):
        words = [w for w in tokens if w not in stopwords.words('english')]
        return words

    def word_lemmatizer(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lem_text = [lemmatizer.lemmatize(i) for i in tokens]
        return lem_text

    def preprocess(self):
        tokens = self.tokenization()
        tokens_sw = tokens.progress_apply(lambda x: self.remove_stopwords(x))
        final_tokens = tokens_sw.progress_apply(
            lambda x: self.word_lemmatizer(x))
        return final_tokens

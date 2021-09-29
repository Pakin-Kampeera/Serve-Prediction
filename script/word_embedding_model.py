from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from script.data_preprocess import Posts
from script.word_embedding_vectorizer import WordEmbeddingVectorizer

output_word_model = open('./artifacts/word_model.pkl', 'rb')
word_model = pickle.load(output_word_model)

# Logistic Regression
output_word_embedding_lg = open('./artifacts/word_embedding_lr.pkl', 'rb')
word_embedding_lg = pickle.load(output_word_embedding_lg)

# Random Forrest
# output_word_embedding_rf = open('./artifacts/word_embedding_rf.pkl', 'rb')
# word_embedding_rf = pickle.load(output_word_embedding_rf)


class Series(BaseModel):
    series: list


async def text_processes(series: Series):
    global preprocess_series
    preprocess_series = Posts(series).preprocess()
    print(preprocess_series)
    input_matrix = WordEmbeddingVectorizer(word_model).fit(
        preprocess_series).transform(preprocess_series)
    return input_matrix


async def predict(series: Series):
    input_matrix = await text_processes(pd.Series(series.series))
    index_list = np.where(np.all(input_matrix == 0, axis=1))[0]
    trim_matrix = input_matrix[~np.all(input_matrix == 0, axis=1)]
    if trim_matrix.size:
        pred_labels = word_embedding_lg.predict(trim_matrix)
        pred_proba = word_embedding_lg.predict_proba(trim_matrix)
        confidence_score = [prob[1] for prob in pred_proba]
        pred_labels = pred_labels.tolist()
    else:
        pred_labels = []
        confidence_score = []
    for i in index_list:
        pred_labels.insert(i, 2)
        confidence_score.insert(i, '-')
    output = pd.DataFrame(
        {'text': series.series, 'confidence_score': confidence_score, 'labels': pred_labels})
    output['labels'] = output['labels'].map(
        {1: 'stress', 0: 'non-stress', 2: 'can\'t tell'})
    output['words'] = preprocess_series
    return output.to_dict(orient='records')

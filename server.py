from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import pickle
from script.word_embedding_vectorizer import WordEmbeddingVectorizer
from script.data_preprocess import Posts

app = FastAPI(redoc_url=None)


class Series(BaseModel):
    series: list


output_word_model = open('artifacts\word_model.pkl', 'rb')
word_model = pickle.load(output_word_model)

# output_word_vectorizer = open('artifacts\word_vectorizer.pkl', 'rb')
# word_vectorizer = pickle.load(output_word_vectorizer)

output_word_embedding_rf = open('artifacts\word_embedding_rf.pkl', 'rb')
word_embedding_rf = pickle.load(output_word_embedding_rf)


@ app.post('/prediction', tags=['Prediction'])
async def Stress_Prediction(series: Series):
    print(series.series)
    input_matrix = await Text_Processes(pd.Series(series.series))
    pred_labels = word_embedding_rf.predict(input_matrix)
    pred_proba = word_embedding_rf.predict_proba(input_matrix)
    confidence_score = [prob[1] for prob in pred_proba]
    output = pd.DataFrame(
        {'text': series.series, 'confidence_score': confidence_score, 'labels': pred_labels})
    output['labels'] = output['labels'].map({1: 'stress', 0: 'non-stress'})
    return output


async def Text_Processes(series):
    preprocess_series = Posts(series).preprocess()
    input_matrix = WordEmbeddingVectorizer(word_model).fit(
        preprocess_series).transform(preprocess_series)
    return input_matrix


if __name__ == '__main__':
    uvicorn.run('server:app', host='127.0.0.1', reload=True)

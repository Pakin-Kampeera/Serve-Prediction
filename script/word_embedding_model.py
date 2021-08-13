class WordEmbeddingModel():
    def preprocess(series):
        word_embedding_rf = WordEmbeddingVectorizer(word2vec)
        input_matrix = word_embedding_rf.fit(
            preprocess_series).transform(preprocess_series)
        return input_matrix

    def predict(series):
        input_matrix = preprocess(pd.Series(series))
        pred_labels = clf.predict(input_matrix)
        pred_proba = clf.predict_proba(input_matrix)
        confidence_score = [prob[1] for prob in pred_proba]
        output = pd.DataFrame(
            {'text': series, 'confidence_score': confidence_score, 'labels': pred_labels})
        output['labels'] = output['labels'].map({1: 'stress', 0: 'non-stress'})
        return output

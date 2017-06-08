# -*- coding: utf-8 -*-
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

# remove punctuation, lowercase, stem
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize)

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

def data_filter(data, columns):
    df = pd.DataFrame(data, index=range(1, len(data)), columns=columns)
    # filter the posts count != 0
    df = df[df.posts_count != 0]
    # filter the non ascii character
    df.index = range(1, len(df) + 1)
    for i in range(1, df.shape[0]):
        if isinstance(df.ix[i, 'slug'], (int, long, float, complex)):
            df = df.drop([i])
        elif is_ascii(df.ix[i, 'slug']) == False:
            df = df.drop([i])

    df.index = range(1, len(df) + 1)
    return df

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


# print cosine_sim('machine_learning', 'machine-learning')

if __name__ == "__main__":
    data = pd.read_csv("tags.csv", sep=";", encoding="utf-8")
    columns = ['id', 'name', 'slug', 'created_at', 'updated_at', 'followers_count', 'posts_count']

    df_filtered = data_filter(data=data, columns=columns)

    number_tags = df_filtered.shape[0]

    df_result = pd.DataFrame(columns=columns)
    counter = 0
    ignore_array = []
    result_dict = {}
    for i in range(1, number_tags - 1):
        if i % 100 == 0:
            df_result.to_csv('result_{0}.csv'.format(i), sep=',', encoding='utf-8')
            with open('result_{}.json'.format(i), 'w') as fp:
                json.dump(result_dict, fp)
        counter = 0
        print "step :", i, df_filtered.ix[i, 'slug']
        if i not in ignore_array:
            temp_array = []
            for j in range(i + 1, number_tags):
                if cosine_sim(df_filtered.ix[i, 'slug'], df_filtered.ix[j, 'slug']) >= 0.5:
                    counter += 1
                    print "same as :", df_filtered.ix[j, 'slug']
                    if counter == 1:
                        df_result = df_result.append(df_filtered.loc[i])
                        temp_array.append(df_filtered.loc[i].to_dict())
                    df_result = df_result.append(df_filtered.loc[j])
                    temp_array.append(df_filtered.loc[j].to_dict())
                    ignore_array.append(j)
                    result_dict[i] = temp_array

    df_result.to_csv('result.csv', sep=',', encoding='utf-8')
    with open('result.json', 'w') as fp:
        json.dump(result_dict, fp)

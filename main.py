import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


data = pd.read_csv("movies.csv")

data['genres'] = data['genres'].apply(lambda x: x.replace(' ', '')).replace(',', ' ')
print(data)
vectorlizer = TfidfVectorizer(ngram_range=(1, 1))
tfidf_matrix = vectorlizer.fit_transform(data['genres'])
# print(vectorlizer.vocabulary_)
# print(len(vectorlizer.vocabulary_))
# print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['name'], columns=data['name'])




result = cosine_sim_df.loc['Mai'].sort_values(ascending=False)[:11]
# filter different Mai
result = result[result.index != 'Mai']
print(result)
# cosine_sim_df.to_csv("cosine_sim.csv", encoding="utf-8")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import uuid

engine = create_engine(
    "postgresql+psycopg2://cineme_db:MgyzzizOWwpf7IoCqHM5n62ZnlGx1Qyn@dpg-d30j88ffte5s73efq5q0-a.singapore-postgres.render.com:5432/cineme_db_muz4"
)



def hybrid_recommendation(user_id, movie_id, top_n=10, alpha=0.5):
    data_reviews = pd.read_sql("SELECT user_id, movie_id, rating FROM reviews", engine)
    query = """
    SELECT m.id,
        string_agg(g.name_en, '|' ORDER BY g.name_en) AS genres
    FROM movies m
    JOIN movie_genres mg ON m.id = mg.movie_id
    JOIN genres g ON mg.genre_id = g.id
    GROUP BY m.id
    """
    data_movies = pd.read_sql(query, engine)

    cbf = content_based_filtering(data_movies, movie_id, top_n=top_n)  # lấy nhiều hơn chút
    cf = collaborative_filtering(data_reviews, user_id, top_n=top_n)
    cf["cf_score"] = cf["cf_score"] / 10.0  



    # Merge hai bảng theo movie_id
    merged = pd.merge(cbf, cf, on="movie_id", how="outer").fillna(0)

    # Tính hybrid score
    merged["hybrid_score"] = alpha * merged["cf_score"] + (1 - alpha) * merged["cbf_score"]

    # Sắp xếp và trả kết quả
    return merged.sort_values("hybrid_score", ascending=False).head(top_n)

def collaborative_filtering(data, user_id, top_n=5):
    # data_review = pd.read_csv("ratings.csv", dtype={"user_id": str, "movie_id": str})
    user_item_matrix = data.pivot_table(index='user_id', columns='movie_id', values='rating')

    user_similarity = cosine_similarity(user_item_matrix.fillna(0))
    user_similarity_df = pd.DataFrame(user_similarity, 
                                      index=user_item_matrix.index, 
                                      columns=user_item_matrix.index)

    sims = user_similarity_df.loc[user_id]
    movies_not_watched = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index

    predictions = []
    for movie in movies_not_watched:
        ratings = user_item_matrix[movie]
        mask = ratings.notna()
        sims_filtered = sims[mask]
        ratings_filtered = ratings[mask]

        if sims_filtered.sum() > 0:
            pred_rating = (sims_filtered * ratings_filtered).sum() / sims_filtered.sum()
            predictions.append({"movie_id": movie, "cf_score": pred_rating})

    return pd.DataFrame(predictions).sort_values("cf_score", ascending=False).head(top_n)

def content_based_filtering(data, movie_id, top_n=5):
    # data = pd.read_csv("movies.csv", dtype={"movie_id": str})
    data['genres'] = data['genres'].apply(lambda x: x.replace(' ', '').replace(',', ' '))
    
    # TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(data['genres'])

    # Cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=data['id'], columns=data['id'])

    # Lấy top phim tương tự
    sim_scores = cosine_sim_df[movie_id].sort_values(ascending=False).iloc[1:top_n+1]  # bỏ chính nó (score=1.0)

    # Convert sang DataFrame
    cbf_df = sim_scores.reset_index()
    cbf_df.columns = ["movie_id", "cbf_score"]

    return cbf_df



# import json

# with open("data.json", "r", encoding="utf-8") as f:
#     data_json = json.load(f)
# df_reviews = pd.DataFrame(data_json['listReview'])
# print(df_reviews)
# result = hybrid_recommendation(df_reviews, '5943eeee-6f24-484c-9b8d-418ead40963e', 'c23d985b-bb1c-475e-a686-271726598345')

# print(result)

user_id = uuid.UUID('5943eeee-6f24-484c-9b8d-418ead40963e')
movie_id = uuid.UUID('c23d985b-bb1c-475e-a686-271726598345')

result = hybrid_recommendation(user_id, movie_id)

print(result)
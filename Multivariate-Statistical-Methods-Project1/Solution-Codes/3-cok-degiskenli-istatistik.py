import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import os
from tqdm import tqdm
from numba import jit, prange
import time
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import concurrent.futures
from multiprocessing import freeze_support

@jit(nopython=True, parallel=False)
def cosine_similarity_numba(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def user_knn_predictor(user_id, movie_id, k):
    global user_item_matrix
    try:
        user_idx = user_id_map[user_id]
        movie_idx = movie_id_map[movie_id]
    except KeyError:
        return train_data['rating'].mean()
    
    if user_item_matrix[user_idx, movie_idx] != 0:
        return user_item_matrix[user_idx, movie_idx]
    
    user_vector = user_item_matrix[user_idx].toarray().flatten()
    similarities = []
    for i in range(user_item_matrix.shape[0]):
        if i == user_idx:
            continue
        
        if user_item_matrix[i, movie_idx] == 0:
            continue
        
        other_vector = user_item_matrix[i].toarray().flatten()
        sim = cosine_similarity_numba(user_vector, other_vector)
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_users = similarities[:k]
    
    if len(top_k_users) == 0:
        return train_data['rating'].mean()
    
    weighted_sum = 0
    similarity_sum = 0
    
    for user_idx, similarity in top_k_users:
        rating = user_item_matrix[user_idx, movie_idx]
        weighted_sum += rating * similarity
        similarity_sum += similarity
    
    if similarity_sum == 0:
        return train_data['rating'].mean()
    
    return weighted_sum / similarity_sum

def item_knn_predictor(user_id, movie_id, k):
    global item_user_matrix, user_item_matrix
    try:
        user_idx = user_id_map[user_id]
        movie_idx = movie_id_map[movie_id]
    except KeyError:
        return train_data['rating'].mean()
    
    if user_item_matrix[user_idx, movie_idx] != 0:
        return user_item_matrix[user_idx, movie_idx]
    
    user_ratings = user_item_matrix[user_idx].toarray().flatten()
    rated_movie_indices = np.where(user_ratings > 0)[0]
    
    if len(rated_movie_indices) == 0:
        return train_data['rating'].mean()
    
    movie_vector = item_user_matrix[movie_idx].toarray().flatten()
    similarities = []
    for i in rated_movie_indices:
        other_vector = item_user_matrix[i].toarray().flatten()
        sim = cosine_similarity_numba(movie_vector, other_vector)
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_movies = similarities[:k]
    
    if len(top_k_movies) == 0:
        return train_data['rating'].mean()
    
    weighted_sum = 0
    similarity_sum = 0
    
    for movie_idx, similarity in top_k_movies:
        rating = user_ratings[movie_idx]
        weighted_sum += rating * similarity
        similarity_sum += similarity
    
    if similarity_sum == 0:
        return train_data['rating'].mean()
    
    return weighted_sum / similarity_sum

def process_row(row, predictor, k):
    user_id = row['userId']
    movie_id = row['movieId']
    true_rating = row['rating']
    
    predicted_rating = predictor(user_id, movie_id, k)
    
    return true_rating, predicted_rating

def evaluate_knn_model_parallel(test_data, model_type, k_values, n_jobs=4):
    results = {}
    
    if model_type == 'user':
        predictor = user_knn_predictor
    else:
        predictor = item_knn_predictor
    
    for k in k_values:
        print(f"Değerlendiriliyor: {model_type.capitalize()}-KNN, K={k}")
        
        import multiprocessing
        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_row = {
                executor.submit(process_row, row, predictor, k): idx 
                for idx, row in test_data.iterrows()
            }
            
            true_ratings = []
            predicted_ratings = []
            
            for future in tqdm(concurrent.futures.as_completed(future_to_row),
                              total=len(future_to_row),
                              desc=f"{model_type.capitalize()}-KNN (K={k})", 
                              ncols=100):
                try:
                    true_rating, predicted_rating = future.result()
                    true_ratings.append(true_rating)
                    predicted_ratings.append(predicted_rating)
                except Exception as e:
                    print(f"Bir işlemde hata oluştu: {e}")
        
        rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
        results[k] = rmse
        print(f"{model_type.capitalize()}-KNN, K={k}, Test RMSE: {rmse:.4f}")
    
    return results

if __name__ == '__main__':
    freeze_support()
    start_time = time.time()

    print("Veri seti yükleniyor...")
    ratings = pd.read_csv("/Users/mrved/Desktop/ml-32m/ratings.csv")

    print("Veri seti boyutu:", ratings.shape)
    print("\nİlk 5 satır:")
    print(ratings.head())

    print("\nVeri seti bilgileri:")
    print(ratings.info())

    print("\nBetimsel istatistikler:")
    print(ratings.describe())

    n_users = ratings['userId'].nunique()
    n_movies = ratings['movieId'].nunique()
    print(f"\nBenzersiz kullanıcı sayısı: {n_users}")
    print(f"Benzersiz film sayısı: {n_movies}")

    print("\nVeri eğitim ve test setlerine ayrılıyor...")
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    print(f"\nEğitim seti boyutu: {train_data.shape}")
    print(f"Test seti boyutu: {test_data.shape}")

    test_sample_size = 1000
    test_sample = test_data.sample(n=test_sample_size, random_state=42)
    print(f"Test örnek boyutu: {test_sample.shape}")

    print("\nSparse matrisler oluşturuluyor...")

    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()

    user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
    movie_id_map = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    rev_user_id_map = {i: user_id for user_id, i in user_id_map.items()}
    rev_movie_id_map = {i: movie_id for movie_id, i in movie_id_map.items()}

    train_mapped = train_data.copy()
    train_mapped['user_idx'] = train_mapped['userId'].map(user_id_map)
    train_mapped['movie_idx'] = train_mapped['movieId'].map(movie_id_map)

    user_item_matrix = csr_matrix(
        (train_mapped['rating'], 
         (train_mapped['user_idx'], train_mapped['movie_idx'])),
        shape=(len(user_ids), len(movie_ids))
    )

    item_user_matrix = user_item_matrix.T.tocsr()

    print(f"User-Item matris boyutu: {user_item_matrix.shape}")
    print(f"Item-User matris boyutu: {item_user_matrix.shape}")
    print(f"Seyreklik oranı: {1.0 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])):.4%}")

    k_values = [5, 10, 20, 30, 40, 50]

    print("\nKullanıcı-temelli KNN değerlendiriliyor (paralel)...")
    user_knn_results = evaluate_knn_model_parallel(test_sample, 'user', k_values)

    print("\nÖğe-temelli KNN değerlendiriliyor (paralel)...")
    item_knn_results = evaluate_knn_model_parallel(test_sample, 'item', k_values)

    best_user_k = min(user_knn_results, key=user_knn_results.get)
    best_item_k = min(item_knn_results, key=item_knn_results.get)

    print("\nEn iyi K değerleri ve RMSE puanları:")
    print(f"User-KNN: K = {best_user_k}, RMSE = {user_knn_results[best_user_k]:.4f}")
    print(f"Item-KNN: K = {best_item_k}, RMSE = {item_knn_results[best_item_k]:.4f}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nToplam çalışma süresi: {execution_time:.2f} saniye ({execution_time/60:.2f} dakika)")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, [user_knn_results[k] for k in k_values], marker='o', linestyle='-', color='blue')
    plt.title('User-KNN: K vs RMSE')
    plt.xlabel('K değeri')
    plt.ylabel('RMSE')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, [item_knn_results[k] for k in k_values], marker='o', linestyle='-', color='red')
    plt.title('Item-KNN: K vs RMSE')
    plt.xlabel('K değeri')
    plt.ylabel('RMSE')
    plt.grid(True)

    plt.tight_layout()

    desktop_path = os.path.expanduser("~/Desktop")
    plt.savefig(os.path.join(desktop_path, "knn_results.png"), dpi=300)
    plt.show()

    results_df = pd.DataFrame({
        'Yöntem': ['User-KNN'] * len(k_values) + ['Item-KNN'] * len(k_values),
        'Hiperparametreler': [f'K = {k}, cosine' for k in k_values] * 2,
        'Test RMSE': [user_knn_results[k] for k in k_values] + [item_knn_results[k] for k in k_values]
    })

    print("\nHiperparametre Optimizasyonu Sonuçları:")
    print(results_df)

    best_results_df = pd.DataFrame({
        'Yöntem': ['User-KNN', 'Item-KNN'],
        'Hiperparametreler': [f'K = {best_user_k}, cosine', f'K = {best_item_k}, cosine'],
        'Test RMSE': [user_knn_results[best_user_k], item_knn_results[best_item_k]]
    })

    print("\nEn İyi Sonuçlar:")
    print(best_results_df)

    results_df.to_csv(os.path.join(desktop_path, "knn_all_results.csv"), index=False)
    best_results_df.to_csv(os.path.join(desktop_path, "knn_best_results.csv"), index=False)

    print("\nSonuçlar masaüstüne kaydedildi:")
    print(f"- Grafik: {os.path.join(desktop_path, 'knn_results.png')}")
    print(f"- En iyi sonuçlar: {os.path.join(desktop_path, 'knn_best_results.csv')}")
    print(f"- Tüm sonuçlar: {os.path.join(desktop_path, 'knn_all_results.csv')}")
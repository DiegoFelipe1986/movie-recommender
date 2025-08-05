"""
Utilidades para el sistema de recomendación de películas
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re

def load_movie_data(file_path=None):
    """
    Cargar datos de películas desde el dataset de MovieLens
    
    Args:
        file_path (str): Ruta al archivo CSV con datos de películas
        
    Returns:
        pd.DataFrame: DataFrame con los datos de películas
    """
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # Cargar dataset de MovieLens
        movies_path = "ml-latest-small 2/movies.csv"
        ratings_path = "ml-latest-small 2/ratings.csv"
        
        if os.path.exists(movies_path):
            movies_df = pd.read_csv(movies_path)
            
            # Extraer año del título
            movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
            movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
            
            # Calcular rating promedio si existe ratings.csv
            if os.path.exists(ratings_path):
                ratings_df = pd.read_csv(ratings_path)
                avg_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
                avg_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
                
                # Unir con películas
                movies_df = movies_df.merge(avg_ratings, on='movieId', how='left')
                movies_df['avg_rating'] = movies_df['avg_rating'].fillna(0)
                movies_df['rating_count'] = movies_df['rating_count'].fillna(0)
            else:
                movies_df['avg_rating'] = 0
                movies_df['rating_count'] = 0
            
            return movies_df
        else:
            # Datos de ejemplo como fallback
            movies_data = {
                'movieId': range(1, 21),
                'title': [
                    'The Shawshank Redemption (1994)', 'The Godfather (1972)', 'Pulp Fiction (1994)',
                    'The Dark Knight (2008)', 'Fight Club (1999)', 'Forrest Gump (1994)',
                    'Inception (2010)', 'The Matrix (1999)', 'Goodfellas (1990)', 'The Silence of the Lambs (1991)',
                    'Interstellar (2014)', 'The Green Mile (1999)', 'The Departed (2006)', 'The Lion King (1994)',
                    'Gladiator (2000)', 'The Prestige (2006)', 'The Usual Suspects (1995)', 'Se7en (1995)',
                    'The Sixth Sense (1999)', 'The Truman Show (1998)'
                ],
                'genres': [
                    'Drama', 'Crime|Drama', 'Crime|Drama', 'Action|Crime|Drama',
                    'Drama', 'Drama|Romance', 'Action|Adventure|Sci-Fi',
                    'Action|Sci-Fi', 'Biography|Crime|Drama', 'Crime|Drama|Thriller',
                    'Adventure|Drama|Sci-Fi', 'Crime|Drama', 'Crime|Drama|Thriller',
                    'Animation|Adventure|Drama', 'Action|Adventure|Drama',
                    'Drama|Mystery|Thriller', 'Crime|Drama|Mystery', 'Crime|Drama|Mystery',
                    'Drama|Mystery|Thriller', 'Comedy|Drama'
                ],
                'year': [1994, 1972, 1994, 2008, 1999, 1994, 2010, 1999, 1990, 1991,
                         2014, 1999, 2006, 1994, 2000, 2006, 1995, 1995, 1999, 1998],
                'avg_rating': [9.3, 9.2, 8.9, 9.0, 8.8, 8.8, 8.8, 8.7, 8.7, 8.6,
                               8.6, 8.6, 8.5, 8.5, 8.5, 8.5, 8.5, 8.6, 8.1, 8.1]
            }
            return pd.DataFrame(movies_data)

def create_similarity_matrix(movies_df, feature_column='genres'):
    """
    Crear matriz de similitud basada en una columna de características
    
    Args:
        movies_df (pd.DataFrame): DataFrame con datos de películas
        feature_column (str): Columna a usar para calcular similitud
        
    Returns:
        np.ndarray: Matriz de similitud coseno
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df[feature_column])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_movie_recommendations(movies_df, movie_title, similarity_matrix, n_recommendations=5):
    """
    Obtener recomendaciones de películas basadas en similitud
    
    Args:
        movies_df (pd.DataFrame): DataFrame con datos de películas
        movie_title (str): Título de la película de referencia
        similarity_matrix (np.ndarray): Matriz de similitud
        n_recommendations (int): Número de recomendaciones a devolver
        
    Returns:
        pd.DataFrame: DataFrame con las películas recomendadas
    """
    try:
        # Buscar película por título (ignorando año)
        movie_title_clean = re.sub(r'\s*\(\d{4}\)', '', movie_title).strip()
        movies_df['title_clean'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        
        # Buscar coincidencia exacta o parcial
        movie_idx = movies_df[movies_df['title_clean'].str.contains(movie_title_clean, case=False, na=False)].index
        
        if len(movie_idx) == 0:
            return pd.DataFrame()
        
        movie_idx = movie_idx[0]
        
        # Obtener similitudes para la película
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Obtener las películas más similares (excluyendo la misma)
        sim_scores = sim_scores[1:n_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        
        return movies_df.iloc[movie_indices]
    except (IndexError, KeyError):
        return pd.DataFrame()

def search_movies(movies_df, query, limit=10):
    """
    Buscar películas por título
    
    Args:
        movies_df (pd.DataFrame): DataFrame con datos de películas
        query (str): Término de búsqueda
        limit (int): Número máximo de resultados
        
    Returns:
        pd.DataFrame: Películas que coinciden con la búsqueda
    """
    query_lower = query.lower()
    matches = movies_df[
        movies_df['title'].str.lower().str.contains(query_lower, na=False)
    ].head(limit)
    return matches

def get_popular_movies(movies_df, min_ratings=10, limit=20):
    """
    Obtener películas populares basadas en calificaciones
    
    Args:
        movies_df (pd.DataFrame): DataFrame con datos de películas
        min_ratings (int): Número mínimo de calificaciones
        limit (int): Número máximo de resultados
        
    Returns:
        pd.DataFrame: Películas populares
    """
    popular = movies_df[
        (movies_df['rating_count'] >= min_ratings) & 
        (movies_df['avg_rating'] > 0)
    ].sort_values('avg_rating', ascending=False).head(limit)
    return popular

def get_movies_by_genre(movies_df, genre, limit=20):
    """
    Obtener películas por género
    
    Args:
        movies_df (pd.DataFrame): DataFrame con datos de películas
        genre (str): Género a buscar
        limit (int): Número máximo de resultados
        
    Returns:
        pd.DataFrame: Películas del género especificado
    """
    genre_movies = movies_df[
        movies_df['genres'].str.contains(genre, case=False, na=False)
    ].sort_values('avg_rating', ascending=False).head(limit)
    return genre_movies

def save_model(model, file_path):
    """
    Guardar modelo entrenado en un archivo
    
    Args:
        model: Modelo a guardar
        file_path (str): Ruta donde guardar el modelo
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    """
    Cargar modelo desde un archivo
    
    Args:
        file_path (str): Ruta del archivo del modelo
        
    Returns:
        Modelo cargado
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_rating_stats(movies_df):
    """
    Calcular estadísticas de calificaciones
    
    Args:
        movies_df (pd.DataFrame): DataFrame con datos de películas
        
    Returns:
        dict: Diccionario con estadísticas
    """
    return {
        'mean_rating': movies_df['avg_rating'].mean(),
        'median_rating': movies_df['avg_rating'].median(),
        'std_rating': movies_df['avg_rating'].std(),
        'min_rating': movies_df['avg_rating'].min(),
        'max_rating': movies_df['avg_rating'].max(),
        'total_movies': len(movies_df),
        'total_ratings': movies_df['rating_count'].sum()
    }

def get_genre_distribution(movies_df):
    """
    Obtener distribución de géneros
    
    Args:
        movies_df (pd.DataFrame): DataFrame con datos de películas
        
    Returns:
        pd.Series: Series con conteo de géneros
    """
    # Separar géneros múltiples
    all_genres = []
    for genres in movies_df['genres']:
        if pd.notna(genres):
            all_genres.extend([genre.strip() for genre in genres.split('|')])
    
    return pd.Series(all_genres).value_counts() 
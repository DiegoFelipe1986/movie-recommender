#!/usr/bin/env python3
"""
Script de prueba para el sistema de recomendación de películas
"""

import sys
import os

# Agregar el directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils import (
    load_movie_data, create_similarity_matrix, get_movie_recommendations,
    search_movies, get_popular_movies, get_movies_by_genre,
    calculate_rating_stats, get_genre_distribution
)

def test_recommendation_system():
    """Probar el sistema de recomendación"""
    print("🎬 Probando Sistema de Recomendación de Películas")
    print("=" * 60)
    
    # Cargar datos
    print("📊 Cargando dataset de MovieLens...")
    movies_df = load_movie_data()
    print(f"✅ Cargadas {len(movies_df):,} películas")
    
    # Mostrar estadísticas básicas
    stats = calculate_rating_stats(movies_df)
    print(f"📈 Estadísticas del dataset:")
    print(f"   • Total de películas: {stats['total_movies']:,}")
    print(f"   • Calificaciones totales: {stats['total_ratings']:,}")
    print(f"   • Calificación promedio: {stats['mean_rating']:.2f}")
    print(f"   • Películas con calificaciones: {len(movies_df[movies_df.get('rating_count', 0) > 0]):,}")
    
    # Crear matriz de similitud
    print("\n🔍 Creando matriz de similitud...")
    similarity_matrix = create_similarity_matrix(movies_df)
    print("✅ Matriz de similitud creada")
    
    # Probar búsqueda
    print("\n🔍 Probando búsqueda de películas...")
    test_searches = ['Toy Story', 'Matrix', 'Star Wars', 'Batman', 'Harry Potter']
    
    for search_term in test_searches:
        results = search_movies(movies_df, search_term, limit=3)
        if not results.empty:
            print(f"   • '{search_term}': {len(results)} resultados encontrados")
        else:
            print(f"   • '{search_term}': No se encontraron resultados")
    
    # Probar recomendaciones
    print("\n🎯 Probando recomendaciones...")
    test_movies = ['Toy Story (1995)', 'The Matrix (1999)', 'Pulp Fiction (1994)']
    
    for movie in test_movies:
        print(f"\n📽️  Película de referencia: {movie}")
        recommendations = get_movie_recommendations(movies_df, movie, similarity_matrix, 3)
        
        if not recommendations.empty:
            print("🎬 Películas recomendadas:")
            for idx, row in recommendations.iterrows():
                year_str = f" ({row['year']})" if pd.notna(row['year']) else ""
                rating_str = f" - ⭐ {row['avg_rating']:.1f}" if row['avg_rating'] > 0 else ""
                print(f"   • {row['title']}{year_str} - {row['genres']}{rating_str}")
        else:
            print("❌ No se encontraron recomendaciones")
    
    # Probar películas populares
    print("\n🏆 Probando películas populares...")
    popular_movies = get_popular_movies(movies_df, min_ratings=100, limit=5)
    if not popular_movies.empty:
        print("Top 5 películas más populares:")
        for idx, row in popular_movies.iterrows():
            year_str = f" ({row['year']})" if pd.notna(row['year']) else ""
            print(f"   • {row['title']}{year_str} - ⭐ {row['avg_rating']:.1f} ({row.get('rating_count', 0)} votos)")
    
    # Probar géneros
    print("\n🎭 Probando búsqueda por géneros...")
    genre_dist = get_genre_distribution(movies_df)
    top_genres = genre_dist.head(5)
    print("Top 5 géneros más populares:")
    for genre, count in top_genres.items():
        print(f"   • {genre}: {count} películas")
    
    # Probar películas por género
    test_genre = "Action"
    genre_movies = get_movies_by_genre(movies_df, test_genre, limit=3)
    if not genre_movies.empty:
        print(f"\n📽️  Top 3 películas de {test_genre}:")
        for idx, row in genre_movies.iterrows():
            year_str = f" ({row['year']})" if pd.notna(row['year']) else ""
            rating_str = f" - ⭐ {row['avg_rating']:.1f}" if row['avg_rating'] > 0 else ""
            print(f"   • {row['title']}{year_str}{rating_str}")
    
    print("\n✅ Prueba completada exitosamente!")
    print("🚀 El sistema está listo para usar con el dataset de MovieLens!")

if __name__ == "__main__":
    test_recommendation_system() 
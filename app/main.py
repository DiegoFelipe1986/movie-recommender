import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import (
    load_movie_data, create_similarity_matrix, get_movie_recommendations,
    search_movies, get_popular_movies, get_movies_by_genre,
    calculate_rating_stats, get_genre_distribution
)

# Configuración de la página con tema claro
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background-color: #ffffff;
    }
    .stButton > button {
        background-color: #ff6b6b;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff5252;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .movie-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 4px solid #ff6b6b;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
        border-radius: 8px;
    }
    /* Arreglar contraste en elementos de Streamlit */
    .stTextInput > div > div > input {
        color: #2c3e50 !important;
        background-color: #ffffff !important;
    }
    .stSelectbox > div > div > div {
        color: #2c3e50 !important;
        background-color: #ffffff !important;
    }
    .stSlider > div > div > div > div {
        color: #2c3e50 !important;
    }
    /* Mejorar contraste en métricas */
    .metric-container .stMetric {
        color: #2c3e50 !important;
    }
    .metric-container .stMetric > div > div {
        color: #2c3e50 !important;
    }
    /* Asegurar que el texto en las tarjetas sea legible */
    .movie-card h4, .movie-card p {
        color: #2c3e50 !important;
    }
    /* Mejorar contraste en el sidebar */
    .css-1d391kg {
        background-color: #f8f9fa !important;
    }
    /* Asegurar que los labels sean legibles */
    .stTextInput > div > div > label {
        color: #2c3e50 !important;
    }
    .stSelectbox > div > div > label {
        color: #2c3e50 !important;
    }
    .stSlider > div > div > label {
        color: #2c3e50 !important;
    }
    /* Forzar colores oscuros solo en el contenido principal */
    .main .stMarkdown {
        color: #2c3e50 !important;
    }
    .main .stMarkdown p {
        color: #2c3e50 !important;
    }
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3, .main .stMarkdown h4, .main .stMarkdown h5, .main .stMarkdown h6 {
        color: #2c3e50 !important;
    }
    .main .stMarkdown ul, .main .stMarkdown ol {
        color: #2c3e50 !important;
    }
    .main .stMarkdown li {
        color: #2c3e50 !important;
    }
    .main .stMarkdown strong {
        color: #2c3e50 !important;
    }
    /* Contenedores específicos */
    .content-container {
        color: #2c3e50 !important;
    }
    .content-container * {
        color: #2c3e50 !important;
    }
    .metric-container {
        color: #2c3e50 !important;
    }
    .metric-container * {
        color: #2c3e50 !important;
    }
    /* Forzar contraste en elementos específicos */
    .main .stMarkdown .stMarkdown {
        color: #2c3e50 !important;
    }
    /* Asegurar que las métricas sean legibles */
    .stMetric > div > div {
        color: #2c3e50 !important;
    }
    .stMetric > div > div > div {
        color: #2c3e50 !important;
    }
    /* Excepciones para elementos que deben ser blancos */
    .gradient-text, .white-text {
        color: white !important;
    }
    .metric-value {
        color: #2c3e50 !important;
    }
    /* NO aplicar a elementos del sidebar */
    .css-1d391kg * {
        color: inherit !important;
    }
    .stSidebar * {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con mejor diseño
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; margin-bottom: 30px;">
    <h1 class="white-text" style="color: white; margin: 0; font-size: 3rem;">🎬 Movie Recommender</h1>
    <p class="white-text" style="color: white; margin: 10px 0 0 0; font-size: 1.2rem;">Descubre tu próxima película favorita</p>
</div>
""", unsafe_allow_html=True)

# Sidebar mejorado
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 20px;">
        <h3 class="white-text" style="color: white; text-align: center; margin: 0;">Navegación</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.selectbox(
        "Selecciona una página:",
        ["🏠 Inicio", "🎯 Recomendaciones", "🔍 Búsqueda", "🏆 Películas Populares", "🎭 Por Género", "📊 Análisis de Datos", "ℹ️ Acerca de"]
    )

# Cargar datos
@st.cache_data
def load_data():
    return load_movie_data()

movies_df = load_data()

# Extraer nombre de página sin emoji
page_name = page.split(" ", 1)[1] if " " in page else page

if page_name == "Inicio":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;">
        <h2 style="color: white; text-align: center; margin: 0;">Bienvenido al Sistema de Recomendación</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background-color: #ffffff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #2c3e50; margin-top: 0;">✨ Características principales:</h3>
            <ul style="color: #34495e; font-size: 1.1rem;">
                <li><strong>🎯 Recomendaciones personalizadas</strong> basadas en géneros y calificaciones</li>
                <li><strong>🔍 Búsqueda avanzada</strong> de películas</li>
                <li><strong>📊 Análisis de datos</strong> con visualizaciones interactivas</li>
                <li><strong>🏆 Películas populares</strong> y por género</li>
                <li><strong>🎨 Interfaz intuitiva</strong> para explorar películas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        stats = calculate_rating_stats(movies_df)
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%); 
                    padding: 20px; border-radius: 15px; color: white;">
            <h3 style="color: white; margin-top: 0;">📊 Estadísticas</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-container" style="background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
        """, unsafe_allow_html=True)
        st.metric("🎬 Películas", f"{stats['total_movies']:,}")
        st.metric("⭐ Calificaciones", f"{stats['total_ratings']:,}")
        st.metric("📈 Promedio", f"{stats['mean_rating']:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Mostrar algunas películas populares
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%); 
                padding: 20px; border-radius: 15px; margin: 30px 0;">
        <h3 style="color: white; margin: 0;">🏆 Películas Más Populares</h3>
    </div>
    """, unsafe_allow_html=True)
    
    popular_movies = get_popular_movies(movies_df, min_ratings=50, limit=10)
    
    if not popular_movies.empty:
        for idx, row in popular_movies.iterrows():
            st.markdown(f"""
            <div class="movie-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <h4 style="color: #2c3e50; margin: 0;">{row['title']}</h4>
                        <p style="color: #7f8c8d; margin: 5px 0;">{row['genres']}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="color: #e74c3c; font-weight: bold; margin: 0;">⭐ {row['avg_rating']:.1f}</p>
                        <p style="color: #95a5a6; font-size: 0.9rem; margin: 0;">({row.get('rating_count', 0)} votos)</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif page_name == "Recomendaciones":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 30px;">
        <h2 style="color: white; text-align: center; margin: 0;">🎯 Obtener Recomendaciones</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Búsqueda de película con mejor diseño
    st.markdown("""
    <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <h4 style="color: #2c3e50; margin: 0 0 10px 0;">🔍 Busca una película para obtener recomendaciones:</h4>
    """, unsafe_allow_html=True)
    search_query = st.text_input(
        "🔍 Busca una película para obtener recomendaciones:",
        placeholder="Ej: Toy Story, The Matrix, Pulp Fiction..."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if search_query:
        search_results = search_movies(movies_df, search_query, limit=10)
        
        if not search_results.empty:
            st.markdown("""
            <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h4 style="color: #27ae60; margin: 0;">✅ Películas encontradas:</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Crear opciones para el selectbox
            options = [f"{row['title']} ({row['year']})" if pd.notna(row['year']) else row['title'] 
                      for idx, row in search_results.iterrows()]
            
            selected_movie = st.selectbox(
                "🎬 Selecciona una película:",
                options
            )
            
            if selected_movie:
                # Extraer título sin año para búsqueda
                movie_title = selected_movie.split(' (')[0] if ' (' in selected_movie else selected_movie
                
                # Mostrar información de la película seleccionada
                movie_info = search_results[search_results['title'].str.contains(movie_title, na=False)].iloc[0]
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%); 
                            padding: 20px; border-radius: 15px; margin: 20px 0;">
                    <h3 style="color: white; margin: 0;">🎬 {movie_info['title']}</h3>
                    <p style="color: white; margin: 10px 0;">{movie_info['genres']}</p>
                    <div style="display: flex; gap: 20px;">
                        <span style="color: white;">📅 {movie_info['year'] if pd.notna(movie_info['year']) else 'N/A'}</span>
                        <span style="color: white;">⭐ {movie_info['avg_rating']:.1f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Obtener recomendaciones
                with st.spinner("🔍 Buscando recomendaciones..."):
                    similarity_matrix = create_similarity_matrix(movies_df)
                    recommendations = get_movie_recommendations(movies_df, movie_title, similarity_matrix, 5)
                
                if not recommendations.empty:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); 
                                padding: 20px; border-radius: 15px; margin: 20px 0;">
                        <h3 style="color: white; margin: 0;">🎬 Películas recomendadas para ti:</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class="movie-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="flex: 1;">
                                    <h4 style="color: #2c3e50; margin: 0;">{row['title']}</h4>
                                    <p style="color: #7f8c8d; margin: 5px 0;">{row['genres']}</p>
                                </div>
                                <div style="text-align: right;">
                                    <p style="color: #e74c3c; font-weight: bold; margin: 0;">⭐ {row['avg_rating']:.1f}</p>
                                    <p style="color: #95a5a6; font-size: 0.9rem; margin: 0;">{row['year'] if pd.notna(row['year']) else 'N/A'}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("❌ No se encontraron recomendaciones para esta película.")
        else:
            st.error("❌ No se encontraron películas con ese nombre.")

elif page_name == "Búsqueda":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 30px;">
        <h2 style="color: white; text-align: center; margin: 0;">🔍 Búsqueda de Películas</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <h4 style="color: #2c3e50; margin: 0 0 10px 0;">🔍 Busca películas por título:</h4>
    """, unsafe_allow_html=True)
    search_query = st.text_input(
        "🔍 Busca películas por título:",
        placeholder="Ej: Star Wars, Harry Potter, Batman..."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if search_query:
        results = search_movies(movies_df, search_query, limit=20)
        
        if not results.empty:
            st.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h4 style="color: #27ae60; margin: 0;">✅ Resultados para '{search_query}':</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, row in results.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <h4 style="color: #2c3e50; margin: 0;">{row['title']}</h4>
                            <p style="color: #7f8c8d; margin: 5px 0;">{row['genres']}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="color: #e74c3c; font-weight: bold; margin: 0;">⭐ {row['avg_rating']:.1f}</p>
                            <p style="color: #95a5a6; font-size: 0.9rem; margin: 0;">{row['year'] if pd.notna(row['year']) else 'N/A'}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ No se encontraron películas con ese nombre.")

elif page_name == "Películas Populares":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 30px;">
        <h2 style="color: white; text-align: center; margin: 0;">🏆 Películas Más Populares</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <h4 style="color: #2c3e50; margin: 0 0 15px 0;">⚙️ Configuración de filtros:</h4>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        min_ratings = st.slider("📊 Mínimo de calificaciones:", 1, 1000, 50)
    with col2:
        limit = st.slider("🎬 Número de películas:", 5, 50, 20)
    st.markdown("</div>", unsafe_allow_html=True)
    
    popular_movies = get_popular_movies(movies_df, min_ratings=min_ratings, limit=limit)
    
    if not popular_movies.empty:
        for idx, row in popular_movies.iterrows():
            st.markdown(f"""
            <div class="movie-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <h4 style="color: #2c3e50; margin: 0;">{row['title']}</h4>
                        <p style="color: #7f8c8d; margin: 5px 0;">{row['genres']}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="color: #e74c3c; font-weight: bold; margin: 0;">⭐ {row['avg_rating']:.1f}</p>
                        <p style="color: #95a5a6; font-size: 0.9rem; margin: 0;">({row.get('rating_count', 0)} votos)</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("❌ No se encontraron películas con los criterios especificados.")

elif page_name == "Por Género":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 30px;">
        <h2 style="color: white; text-align: center; margin: 0;">🎭 Películas por Género</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Obtener géneros únicos
    genre_dist = get_genre_distribution(movies_df)
    available_genres = genre_dist.index.tolist()
    
    st.markdown("""
    <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <h4 style="color: #2c3e50; margin: 0 0 15px 0;">⚙️ Configuración de filtros:</h4>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        selected_genre = st.selectbox("🎭 Selecciona un género:", available_genres)
    with col2:
        limit = st.slider("🎬 Número de películas:", 5, 50, 20)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if selected_genre:
        genre_movies = get_movies_by_genre(movies_df, selected_genre, limit=limit)
        
        if not genre_movies.empty:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%); 
                        padding: 20px; border-radius: 15px; margin: 20px 0;">
                <h3 style="color: white; margin: 0;">🎭 Películas de {selected_genre}:</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, row in genre_movies.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <h4 style="color: #2c3e50; margin: 0;">{row['title']}</h4>
                            <p style="color: #7f8c8d; margin: 5px 0;">{row['genres']}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="color: #e74c3c; font-weight: bold; margin: 0;">⭐ {row['avg_rating']:.1f}</p>
                            <p style="color: #95a5a6; font-size: 0.9rem; margin: 0;">{row['year'] if pd.notna(row['year']) else 'N/A'}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"❌ No se encontraron películas del género {selected_genre}.")

elif page_name == "Análisis de Datos":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 30px;">
        <h2 style="color: white; text-align: center; margin: 0;">📊 Análisis de Datos</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Estadísticas generales
    stats = calculate_rating_stats(movies_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%); 
                    padding: 15px; border-radius: 10px; text-align: center; color: white;">
            <h3 style="margin: 0;">🎬</h3>
            <h2 style="margin: 5px 0;">{stats['total_movies']:,}</h2>
            <p style="margin: 0;">Películas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); 
                    padding: 15px; border-radius: 10px; text-align: center; color: white;">
            <h3 style="margin: 0;">⭐</h3>
            <h2 style="margin: 5px 0;">{stats['mean_rating']:.2f}</h2>
            <p style="margin: 0;">Promedio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%); 
                    padding: 15px; border-radius: 10px; text-align: center; color: white;">
            <h3 style="margin: 0;">📊</h3>
            <h2 style="margin: 5px 0;">{stats['total_ratings']:,}</h2>
            <p style="margin: 0;">Calificaciones</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        rated_movies = len(movies_df[movies_df.get('rating_count', 0) > 0])
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); 
                    padding: 15px; border-radius: 10px; text-align: center; color: white;">
            <h3 style="margin: 0;">✅</h3>
            <h2 style="margin: 5px 0;">{rated_movies:,}</h2>
            <p style="margin: 0;">Con calificaciones</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de calificaciones
        ratings_with_votes = movies_df[movies_df.get('rating_count', 0) > 0]
        if not ratings_with_votes.empty:
            fig_ratings = px.histogram(
                ratings_with_votes, 
                x='avg_rating', 
                nbins=20,
                title="Distribución de Calificaciones",
                labels={'avg_rating': 'Calificación Promedio', 'count': 'Número de Películas'},
                color_discrete_sequence=['#ff6b6b']
            )
            fig_ratings.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#2c3e50',
                font_color='#2c3e50'
            )
            fig_ratings.update_xaxes(
                title_font_color='#2c3e50',
                tickfont_color='#2c3e50'
            )
            fig_ratings.update_yaxes(
                title_font_color='#2c3e50',
                tickfont_color='#2c3e50'
            )
            st.plotly_chart(fig_ratings, use_container_width=True)
    
    with col2:
        # Películas por año
        movies_with_year = movies_df[movies_df['year'].notna()]
        if not movies_with_year.empty:
            fig_years = px.scatter(
                movies_with_year,
                x='year',
                y='avg_rating',
                title="Calificaciones por Año",
                labels={'year': 'Año', 'avg_rating': 'Calificación Promedio'},
                color_discrete_sequence=['#667eea']
            )
            fig_years.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#2c3e50',
                font_color='#2c3e50'
            )
            fig_years.update_xaxes(
                title_font_color='#2c3e50',
                tickfont_color='#2c3e50'
            )
            fig_years.update_yaxes(
                title_font_color='#2c3e50',
                tickfont_color='#2c3e50'
            )
            st.plotly_chart(fig_years, use_container_width=True)
    
    # Top géneros
    st.markdown("""
    <div style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); 
                padding: 20px; border-radius: 15px; margin: 30px 0;">
        <h3 style="color: white; margin: 0;">🎭 Géneros Más Populares</h3>
    </div>
    """, unsafe_allow_html=True)
    
    genre_dist = get_genre_distribution(movies_df)
    top_genres = genre_dist.head(15)
    
    fig_genres = px.bar(
        x=top_genres.values,
        y=top_genres.index,
        orientation='h',
        title="Top 15 Géneros",
        labels={'x': 'Número de Películas', 'y': 'Género'},
        color_discrete_sequence=['#ff6b6b']
    )
    fig_genres.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color='#2c3e50',
        font_color='#2c3e50'
    )
    fig_genres.update_xaxes(
        title_font_color='#2c3e50',
        tickfont_color='#2c3e50'
    )
    fig_genres.update_yaxes(
        title_font_color='#2c3e50',
        tickfont_color='#2c3e50'
    )
    st.plotly_chart(fig_genres, use_container_width=True)

elif page_name == "Acerca de":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 30px;">
        <h2 style="color: white; text-align: center; margin: 0;">ℹ️ Acerca del Proyecto</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="content-container" style="background-color: #ffffff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎬 Movie Recommender")
        
        st.markdown("""
        Este proyecto implementa un sistema de recomendación de películas utilizando técnicas de machine learning.
        """)
        
        st.markdown("#### 📊 Dataset:")
        st.markdown("""
        - **MovieLens**: Dataset con 9,744 películas y calificaciones de usuarios
        - **Fuente**: GroupLens Research Group, University of Minnesota
        """)
        
        st.markdown("#### 🛠️ Tecnologías utilizadas:")
        st.markdown("""
        - **Python**: Lenguaje principal
        - **Streamlit**: Framework para la interfaz web
        - **Scikit-learn**: Algoritmos de machine learning
        - **Pandas**: Manipulación de datos
        - **Plotly**: Visualizaciones interactivas
        """)
        
        st.markdown("#### 🎯 Algoritmos implementados:")
        st.markdown("""
        - **Filtrado colaborativo**: Basado en similitud entre usuarios
        - **Análisis de contenido**: Basado en características de las películas
        - **TF-IDF**: Para análisis de géneros
        """)
        
        st.markdown("#### ✨ Características:")
        st.markdown("""
        - Búsqueda avanzada de películas
        - Recomendaciones personalizadas
        - Análisis de datos y visualizaciones
        - Películas populares y por género
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer mejorado
st.markdown("""
<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
            padding: 20px; border-radius: 15px; margin-top: 50px; text-align: center;">
    <p style="color: white; margin: 0;">© 2024 Movie Recommender - Sistema de recomendación de películas</p>
</div>
""", unsafe_allow_html=True) 
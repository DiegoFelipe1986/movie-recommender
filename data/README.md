# Directorio de Datos

Este directorio contiene los archivos de datos utilizados por el sistema de recomendación de películas.

## Archivos

### `sample_movies.csv`
Archivo CSV con datos de ejemplo de películas que incluye:
- `movie_id`: Identificador único de la película
- `title`: Título de la película
- `genre`: Género(s) de la película
- `year`: Año de lanzamiento
- `rating`: Calificación promedio (1-10)
- `director`: Director de la película
- `cast`: Actores principales
- `description`: Breve descripción de la trama

## Uso

Los datos se cargan automáticamente en la aplicación Streamlit. Si deseas usar tus propios datos:

1. Reemplaza `sample_movies.csv` con tu archivo de datos
2. Asegúrate de que tu archivo tenga las columnas requeridas
3. Modifica la función `load_movie_data()` en `app/utils.py` si es necesario

## Formato de Datos

El sistema espera que los datos estén en formato CSV con las siguientes columnas mínimas:
- `title`: Título de la película
- `genre`: Género(s) separados por comas
- `year`: Año de lanzamiento
- `rating`: Calificación numérica

## Notas

- Los géneros múltiples deben estar separados por comas
- Las calificaciones deben estar en escala de 1-10
- Los años deben ser números enteros de 4 dígitos 
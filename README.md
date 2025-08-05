# Movie Recommender

Sistema de recomendación de películas basado en machine learning.

## Descripción

Este proyecto implementa un sistema de recomendación de películas utilizando técnicas de machine learning como filtrado colaborativo y análisis de contenido.

## Estructura del Proyecto

```
movie-recommender/
├── app/                 # Aplicación Streamlit
├── data/               # Datos y datasets
├── ml-latest-small 2/  # Dataset MovieLens
├── requirements.txt    # Dependencias de Python
├── README.md          # Este archivo
└── .gitignore         # Archivos a ignorar por Git
```

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/DiegoFelipe1986/movie-recommender.git
cd movie-recommender
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación:
```bash
streamlit run app/main.py
```

Para probar el sistema:
```bash
python test_recommender.py
```

## Características

- **Sistema de recomendación** basado en filtrado colaborativo
- **Análisis de contenido** de películas
- **Interfaz web interactiva** con Streamlit
- **Visualizaciones de datos** con Plotly
- **Dataset MovieLens** con 9,744 películas
- **Búsqueda avanzada** de películas
- **Análisis por géneros** y popularidad

## Tecnologías Utilizadas

- **Python**: Lenguaje principal
- **Streamlit**: Framework para la interfaz web
- **Scikit-learn**: Algoritmos de machine learning
- **Pandas**: Manipulación de datos
- **Plotly**: Visualizaciones interactivas

## Dataset

- **MovieLens**: Dataset con 9,744 películas y calificaciones de usuarios
- **Fuente**: GroupLens Research Group, University of Minnesota

## Licencia

MIT

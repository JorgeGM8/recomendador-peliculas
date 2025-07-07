import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from keras.saving import register_keras_serializable
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import pandas as pd

@register_keras_serializable(package='Custom')
class MatrixFactorization(keras.Model):
    """
    Modelo de recomendación que realiza una matriz de factorización utilizando embeddings para usuarios y películas.

    Se añade un sesgo individual.

    Parameters
    ----------
    n_users : int
        Número total de usuarios (tokens únicos).
    n_movies : int
        Número total de películas (tokens únicos).
    embedding_size : int
        Dimensión del vector de embedding para usuarios y películas.
    """
    def __init__(self, n_users:int, n_movies: int, embedding_size: int=10, reg_l2: float=1e-6, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)

        # Definición de variables
        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_size = embedding_size
        self.reg_l2 = reg_l2

        # Embedding de usuarios y de sesgo
        self.user_embedding = layers.Embedding(
            n_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(reg_l2),
        )
        self.user_bias = layers.Embedding(n_users, 1)
        
        # Embedding de películas y de sesgo
        self.movie_embedding = layers.Embedding(
            n_movies,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(reg_l2),
        )
        self.movie_bias = layers.Embedding(n_movies, 1)

    def call(self, inputs:tf.Tensor):
        """
        Calcula la predicción del rating esperado entre un usuario y una película.

        Parameters
        ----------
        inputs : tf.Tensor of shape (batch_size, 2)
            Tensor que contiene pares de [user_id_token, movie_id_token].
            Cada fila representa un usuario y una película para los cuales se
            desea predecir el rating.

        Returns
        -------
        tf.Tensor of shape (batch_size, 1)
            Tensor con las predicciones de ratings, calculadas como el 
            producto escalar entre los embeddings de usuario y película,
            más sus respectivos sesgos (bias).
        """
        # Embedding de inputs de usuarios + sesgo
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])

        # Embedding de inputs de películas + sesgo
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])

        # Cálculo del producto
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

        # Cálculo de predicción
        producto = dot_user_movie + user_bias + movie_bias

        # Ajuste a valores entre 0.5 y 5
        x = 0.5 + 4.5 * tf.sigmoid(producto)

        return x
    
    # Funciones para aplicar configuración al guardar y cargar el modelo
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_users": self.n_users,
            "n_movies": self.n_movies,
            "embedding_size": self.embedding_size,
            "reg_l2": self.reg_l2,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def codificar_usuario(user_id:int, user_to_idx:dict, desconocido:int=-1):
    """
    Codifica un ID de usuario a un índice consecutivo, usando el diccionario correspondiente.

    Transforma en -1 si no existe en el diccionario.

    Parameters
    ----------
    user_id : int
        ID de usuario.
    
    user_to_idx : dict
        Diccionario codificador.
    
    desconocido : int
        Valor asignado a desconocidos (por defecto, -1).

    Returns
    -------
    user_idx : int
        ID codificado del usuario.
    """
    
    user_idx = user_to_idx.get(user_id, desconocido)

    return user_idx


def descodificar_usuario(user_idx:int, idx_to_user:dict):
    """
    Descodifica un índice a un ID de usuario, usando el diccionario correspondiente.

    Parameters
    ----------
    user_idx : int
        Índice o código.
    
    idx_to_user : dict
        Diccionario descodificador.

    Returns
    -------
    user_id : int
        ID del usuario.
    """
    
    user_id = idx_to_user.get(user_idx, None)

    return user_id


def codificar_pelicula(movie_id:int, movie_to_idx:dict, desconocido:int=-1):
    """
    Codifica un ID de película a un índice consecutivo, usando el diccionario correspondiente.

    Transforma en -1 si no existe en el diccionario.

    Parameters
    ----------
    movie_id : int
        ID de película.
    
    movie_to_idx : dict
        Diccionario codificador.
    
    desconocido : int
        Valor asignado a desconocidos (por defecto, -1).

    Returns
    -------
    movie_idx : int
        ID codificado de la película.
    """
    
    movie_idx = movie_to_idx.get(movie_id, desconocido)

    return movie_idx


def descodificar_pelicula(movie_idx:int, idx_to_movie:dict):
    """
    Descodifica un índice a un ID de película, usando el diccionario correspondiente.

    Parameters
    ----------
    movie_idx : int
        Índice o código.
    
    idx_to_movie : dict
        Diccionario descodificador.

    Returns
    -------
    movie_id : int
        ID de la película.
    """
    
    movie_id = idx_to_movie.get(movie_idx, None)

    return movie_id


def revisar_match_llm_1(texto_usuario:str, pelicula:pd.Series, GROQ_API_KEY:str, MODEL_NAME:str):
    """
    LLM que lee la descripción del usuario y lo coteja con la información de cada película.
    
    Devuelve información sobre si coincide o no con lo que el usuario busca.

    Parameters
    ----------
    texto_usuario : str
        Descripción del usuario.
    
    pelicula : Pandas Series
        Serie de Pandas que incluye título, géneros y sinopsis.

    GROQ_API_KEY : str
        API key de Groq.
    
    MODEL_NAME : str
        Modelo elegido disponible en Groq Cloud.

    Returns
    -------
    response.content.strip() : str
        Respuesta del modelo (Sí o No).
    """
    
    llm_1 = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
    )
    
    prompt = f"""
El usuario escribió: '{texto_usuario}'

Revisa lo que dice el usuario sobre lo que quiere ver, teniendo en cuenta si lo que menciona es algo que le gusta o que no le gusta.
Si el usuario quiere ver cualquier película excepto alguna en concreto, ten en cuenta que cualquier película que no incluya esa excepción coincide.
¿Esta película coincide con lo que busca? Solo responde 'Sí' o 'No'.

Título: {pelicula['titulo']}
Géneros: {pelicula['generos']}
Sinopsis: {pelicula['sinopsis']}
"""
    try:
        response = llm_1.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    except Exception as e:
        print(f"Error con Groq: {e}")
        return "Error"


def elegir_pelicula_llm_2(texto_usuario:str, df_predicciones_top:pd.DataFrame, GROQ_API_KEY:str, MODEL_NAME:str):
    """
    LLM que lee la descripción del usuario y las películas que coinciden, y recomienda una basándose en rating, sinopsis y géneros.

    Parameters
    ----------
    texto_usuario : str
        Descripción del usuario.
    
    df_predicciones_top : Pandas DataFrame
        Dataframe de Pandas que incluye rating, título, géneros y sinopsis de las películas que coinciden.

    GROQ_API_KEY : str
        API key de Groq.
    
    MODEL_NAME : str
        Modelo elegido disponible en Groq Cloud.

    Returns
    -------
    response.content.strip() : str
        Respuesta del modelo (recomendación y explicación).
    """

    llm_2 = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
    )
    
    df_coincidencias = df_predicciones_top[df_predicciones_top['coincide'] == True].copy()

    df_coincidencias['contenido'] = df_coincidencias.apply(
        lambda row: f"Título: {row['titulo']}. Géneros: {row['generos']}. Rating predicho: {row['rating']}. Sinopsis: {row['sinopsis']}",
        axis=1
    )
    
    prompt = f"""
El usuario escribió: '{texto_usuario}'

Revisa lo que dice el usuario sobre lo que quiere ver, teniendo en cuenta si lo que menciona es algo que le gusta o que no le gusta.
Si el usuario quiere ver cualquier película excepto alguna en concreto, ten en cuenta que cualquier película que no incluya esa excepción coincide con lo que quiere ver.

El rating es predicho. Basándote en ese rating (siendo 0.5 el peor y 5 el mejor), y en base a la sinopsis y los géneros, decide una película entre las que hay aquí.
Recomienda esa película al usuario y da una explicación breve sobre por qué recomiendas esa película (máximo 2 líneas).
En el caso de basarte principalmente en el rating, no menciones explícitamente el valor, di que crees que es la que más podría gustarle.
Dirígete directamente al usuario.

Las películas son las siguientes:
{df_coincidencias['contenido'].to_list()}
"""
    try:
        response = llm_2.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    except Exception as e:
        print(f"Error con Groq: {e}")
        return "Error"


def elegir_pelicula_llm_3(texto_usuario:str, GROQ_API_KEY:str, MODEL_NAME:str):
    """
    LLM que lee la descripción del usuario y recomienda una película genérica en base a lo que busca.

    Parameters
    ----------
    texto_usuario : str
        Descripción del usuario.

    GROQ_API_KEY : str
        API key de Groq.
    
    MODEL_NAME : str
        Modelo elegido disponible en Groq Cloud.

    Returns
    -------
    response.content.strip() : str
        Respuesta del modelo (recomendación y explicación).
    """

    llm_3 = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
    )

    prompt = f"""
El usuario escribió: '{texto_usuario}'

Revisa lo que dice el usuario sobre lo que quiere ver, teniendo en cuenta si lo que menciona es algo que le gusta o que no le gusta.
Si el usuario quiere ver cualquier película excepto alguna en concreto, ten en cuenta que cualquier película que no incluya esa excepción coincide con lo que quiere ver.

Basándote en tu propio criterio y en lo que quiere el usuario, decide una película entre las que conozcas le pueda gustar.
Recomienda esa película al usuario y da una explicación breve sobre por qué recomiendas esa película (máximo 2 líneas).
Dirígete directamente al usuario.
"""
    
    try:
        response = llm_3.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    except Exception as e:
        print(f"Error con Groq: {e}")
        return "Error"
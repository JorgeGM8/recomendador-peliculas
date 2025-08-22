# Recomendador de películas

## Descripción

Este proyecto consiste en la creación de un sistema de recomendación basado en filtro colaborativo. En este caso, dicho filtro es elaborado con una red neuronal MLP.
Posteriormente, se utilizan tres NLP para procesar las peticiones del usuario y generar una recomendación basada en las preferencias y en las predicciones.

## Instrucciones

Se puede leer el notebook directamente descargando y abriendo en un navegador el archivo html.

Para poder ejecutar el código, se requiere de un programa adecuado para abrir notebooks de Python, como Jupyter o VS Code.
Para instalar las dependencias necesarias, se puede utilizar el archivo `requirements.txt`, con el comando `pip install -r requirements.txt`, o su equivalente en Conda.
También se puede crear el entorno usando UV, con los archivos disponibles `uv.lock` y `pyproject.toml`.

Es necesario crear variables de entorno para la ejecución de código en archivo `.env` (el propio programa las pedirá en el momento si no se encuentran):
- `SYS_PATH`: ruta absoluta del repositorio.
- `OMDB_API_KEY`: clave API para hacer consultas en la base de datos de OMDB.
- `GROQ_API_KEY`: clave API para ejecutar modelos de lenguaje con Groq.

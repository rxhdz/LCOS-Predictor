# Predictor del Síndrome de Bajo Gasto Cardíaco

## Descripción
Este proyecto se enfoca en la identificación de factores de riesgo pronóstico para el Síndrome de Bajo Gasto Cardíaco 
en pacientes sometidos a revascularización miocárdica en el Hospital Universitario Cardiocentro Ernesto Che Guevara 
(Villa Clara, Cuba), utilizando técnicas de inteligencia artificial (IA). El objetivo principal del estudio es 
desarrollar un modelo capaz de predecir si un paciente presentará el síndrome y, una vez desarrollado, extraer los 
factores más relevantes en dicha predicción.

## Tabla de Contenidos
* [Instalación](#instalación)
* [Uso](#uso)
* [Notebooks](#notebooks)
* [Entrenamiento del Modelo](#entrenamiento-del-modelo)
* [Evaluación del Modelo](#evaluación-del-modelo)
* [Explicabilidad](#explicabilidad)
* [Licencia](#licencia)

## Instalación

### Requisitos Previos
- **Git**: Para clonar este repositorio, se debe contar con Git instalado. 
Puede descargarse desde [git-scm.com](https://git-scm.com/).
- **Anaconda**: Se requiere tener instalado Anaconda o Miniconda. Está disponible en 
[sitio oficial de Anaconda](https://www.anaconda.com/products/distribution#download-section).
- **Python**: El proyecto es compatible con la versión 3.12.7 de Python.

### Configuración
0. Clonar el repositorio:
   ```bash
   git clone https://github.com/rxhdz/LCOS-Predictor.git
   cd LCOS-Predictor
   ```

### Usando `environment.yml`
1. Crear un nuevo entorno de conda utilizando el archivo `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```
2. Activar el entorno de conda:
   ```bash
   conda activate low-cardiac-output-syndrome-predictor
   ```

### Usando `requirements.txt` (Opcional)
Si se prefiere crear el entorno virtual manualmente o si se desea instalar paquetes adicionales, puede utilizarse el 
archivo `requirements.txt`:

1. Crear un nuevo entorno de conda:
   ```bash
   conda create --name nombre_de_entorno python=3.12.7
   ```
2. Activar el entorno
   ```bash
   conda activate nombre_de_entorno  # Reemplazar por el nombre preferido
   ```
3. Instalar los paquetes necesarios:
   ```bash
   pip install -r requirements.txt
   ```

### Verificar la Instalación
Para verificar que el entorno se ha configurado correctamente, se puede ejecutar:
```bash
conda list  # Muestra todos los paquetes instalados en el entorno
```

## Uso
Para ejecutar los notebooks, se debe abrir Jupyter Notebook o JupyterLab y navegar hasta el directorio del proyecto. 
Esto puede hacerse con el siguiente comando:
```bash
jupyter notebook
```

Este comando abrirá el navegador y mostrará la vista de árbol de Jupyter con el contenido del directorio. Luego, se 
debe abrir el archivo notebook deseado.

## Notebooks
El proyecto contiene los siguientes notebooks de Jupyter:
- `01_exploratory_data_analysis.ipynb`: Pasos de preprocesamiento, análisis exploratorio de datos y visualizaciones.
- `02_model_training.ipynb`: Entrenamiento de modelos y ajuste de hiperparámetros.
- `03_model_evaluation.ipynb`: Evaluación de modelos, comparación y métricas de rendimiento.
- `04_explainability.ipynb`: Explicación de resultados del modelo utilizando SHAP.

## Entrenamiento del Modelo
El código correspondiente al entrenamiento de los modelos se encuentra en el notebook `02_model_training.ipynb`. 
Incluye:
- Selección de modelos
- Ajuste de hiperparámetros

## Evaluación del Modelo
La evaluación de los modelos se realiza en el notebook `03_model_evaluation.ipynb`. Incluye:
- Comparación entre modelos
- Visualización del rendimiento de los modelos
- Visualización de la importancia de las variables

## Explicabilidad
La explicación detallada de los resultados del modelo se presenta en el notebook `04_explainability.ipynb`. Incluye:
- Gráfico resumen para visualizar la contribución global de cada variable a la predicción
- Gráfico de fuerza para una instancia de cada clase, con el fin de proporcionar explicabilidad local
- Gráficos de dependencia de las variables más importantes

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Para más detalles, consultar el archivo [LICENSE](LICENSE).

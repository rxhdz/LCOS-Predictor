import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def get_mi_scores(X_, y_):
    """
    Calcula y ordena los puntajes de información mutua.

    :param X_: DataFrame que contiene las variables predictoras.

    :param y_: Variable objetivo. Puede ser un array de NumPy o una Serie de pandas con la misma cantidad de filas que `X_`.

    :returns: Serie con los puntajes de información mutua para cada variable predictora, indexada por nombre de variable
        y ordenada en orden descendente (de mayor a menor relevancia).
    """

    mi_scores_ = mutual_info_classif(X_, y_, random_state=42)
    mi_scores_ = pd.Series(mi_scores_, name="mi_scores", index=X_.columns)
    mi_scores_.sort_values(ascending=False, inplace=True)

    return mi_scores_


def get_metrics(model, X_, y_):
    """
    Calcula y retorna métricas de validación cruzada para un modelo de clasificación dado.

    :param model: Modelo de clasificación previamente entrenado o configurado, que debe tener el atributo `feature_names_in_`.

    :param X_: DataFrame con las variables predictoras. Debe contener las columnas indicadas en `model.feature_names_in_`.

    :param y_: Variable objetivo. Puede ser una Serie de pandas o un array de NumPy con la misma longitud que el número de filas en `X_`.

    :returns: Lista de valores flotantes que representa el promedio de cada métrica en el siguiente orden:
        [accuracy, recall, roc_auc, f1].
    """

    features = model.feature_names_in_

    return [
        cross_val_score(model, X_[features], y_, cv=5, scoring="accuracy").mean(),  # Exactitud promedio
        cross_val_score(model, X_[features], y_, cv=5, scoring="recall").mean(),  # Sensibilidad promedio
        cross_val_score(model, X_[features], y_, cv=5, scoring="roc_auc").mean(),  # AUC-ROC promedio
        cross_val_score(model, X_[features], y_, cv=5, scoring="f1").mean(),  # Puntaje F1 promedio
    ]


def get_tuned_model(model, X_, y_, criteria="mean"):
    """
    Ajusta un modelo utilizando únicamente las variables más relevantes según un criterio de selección automática.

    :param model: Modelo base de clasificación o regresión que debe tener implementado el método `fit` y el atributo `feature_importances_` o coeficientes.

    :param X_: DataFrame que contiene las variables predictoras.

    :param y_: Variable objetivo. Puede ser un array de NumPy o una Serie de pandas con la misma cantidad de filas que `X_`.

    :param criteria: Umbral para la selección de características. Puede ser:
        - Un valor numérico (float) para definir un umbral absoluto.
        - Una cadena como 'mean' o 'median', en cuyo caso se usa la media o mediana de las importancias.
        Nota: Por defecto se utilizará la media

    :returns: Una tupla con dos elementos:
        - tuned_model: el modelo entrenado con el subconjunto óptimo de características.
        - feature_names: un array de nombres de las variables seleccionadas.
    """

    params = model.get_params()
    model_ = model.__class__(**params)

    selector = SelectFromModel(estimator=model_, threshold=criteria)
    selector.fit(X_, y_)

    X_selected = selector.transform(X_)
    feature_names = X_.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=feature_names)

    tuned_model = model_.fit(X_selected_df, y_)

    return tuned_model, feature_names


def best_model_per_grid(model, p_grid, X_, y_, criteria="accuracy"):
    """
    Realiza un GridSearch para encontrar la mejor combinación de hiperparámetros
    para un modelo dado, utilizando validación cruzada y una métrica específica como criterio de evaluación.

    :param model: Modelo base que se desea ajustar. Debe ser compatible con `GridSearchCV` (es decir, tener `.fit` y soportar hiperparámetros definidos en `p_grid`).

    :param p_grid: Diccionario que define los hiperparámetros y sus posibles valores para explorar.

    :param X_: DataFrame que contiene las variables predictoras.

    :param y_: Variable objetivo, en formato array de NumPy o Serie de pandas.

    :param criteria: Métrica utilizada para evaluar el desempeño del modelo durante la validación cruzada.
    Puede ser cualquier métrica válida aceptada por `scoring` en `GridSearchCV`

    :returns:  El modelo ajustado con la mejor combinación de hiperparámetros encontrada.
    """

    grid_search = GridSearchCV(model, p_grid, cv=5, scoring=criteria)
    grid_search.fit(X_, y_)

    return grid_search.best_estimator_


def plot_mi_scores(mi_scores_, save=0):
    """
    Genera, muestra y opcionalmente guarda un gráfico de barras horizontal con los puntajes de información mutua
    de las variables predictoras respecto a la variable objetivo.

    Solo se grafican las características cuyo puntaje es mayor que cero.

    :param mi_scores_: Serie que contiene los puntajes de información mutua para cada variable, indexada por nombre de variable.

    :param save: int o bool, opcional (por defecto = 0). Si es distinto de cero o `True`, guarda la figura en la ruta especificada por `save_name`.
    """

    mi_scores = mi_scores_[mi_scores_ > 0]

    plt.figure(figsize=(10, 6))

    sns.barplot(x=mi_scores.values, y=mi_scores.index, orient='h')

    plt.title("Valores de información mutua")
    plt.xlabel("Puntuación")
    plt.ylabel("Atributo")
    plt.tight_layout()

    if save:
        plt.savefig("images/figures/mi_scores.png")

    plt.show()


def plot_distribution(dist, save=0):
    """
    Genera, muestra y opcionalmente guarda un histograma con estimación de densidad (KDE) para visualizar la distribución de una variable categórica binaria.

    :param dist: Distribución de valores categóricos (por ejemplo, 0 y 1) que representan dos grupos distintos en el diagnóstico.
         Se espera que los valores sean binarios (0 y 1), donde:
            - 0 representa ausencia de la condición (ej. no presentó LCOS),
            - 1 representa presencia de la condición (ej. presentó LCOS).

    :param save: int o bool, opcional (por defecto = 0). Si es distinto de cero o `True`, guarda el gráfico en la ruta especificada por `save_name`.
    """

    plt.figure(figsize=(10, 6))

    sns.histplot(dist, kde=True, color="steelblue")

    plt.title("Distribución de los diagnósticos")
    plt.xlabel("Diagnóstico")
    plt.ylabel("Número de muestras")
    plt.xticks(ticks=[0, 1], labels=["Presentó SBGC", "No presentó SBGC"])

    if save:
        plt.savefig("images/figures/distribution.png")

    plt.show()


def plot_feature_importance(model, save=0):
    """
    Genera un gráfico de barras horizontal con la importancia de cada variable predictora,
    según los valores proporcionados por el modelo entrenado.

    :param model: Modelo entrenado que debe tener los atributos `feature_importances_` y `feature_names_in_`.
            Comúnmente se aplica a modelos como RandomForest, GradientBoosting, etc.

    :param title: Título del gráfico. También se utiliza para nombrar el archivo si se guarda.

    :param save: int o bool, opcional (por defecto = 0). Si es distinto de cero o `True`, guarda el gráfico en la carpeta `images/figures/`
            con el nombre derivado del título (`title`), en minúsculas y con espacios reemplazados por guiones bajos.
    """

    importances = model.feature_importances_

    feature_names = model.feature_names_in_

    indices = np.argsort(importances)

    title = f"Importancia de cada atributo para {model.__class__.__name__}"
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.subplots_adjust(left=0.20)
    plt.xlabel("Puntuación")

    if save:
        plt.savefig(f"images/figures/{model.__class__.__name__}_feature_importance.png".lower())

    plt.show()


def plot_cross_val_performance(model, X_, y_, save=0):
    """
    Genera un gráfico de línea que muestra el rendimiento del modelo (precisión) en cada una de las particiones
    de una validación cruzada con 5 particiones.

    :param model: Modelo de clasificación entrenado o preparado para evaluación, que debe contener el atributo `feature_names_in_`.

    :param X_: Conjunto de datos de entrada con las variables predictoras.

    :param y_: Variable objetivo. Puede ser una Serie de pandas o un array de NumPy.

    :param save: int o bool, opcional (por defecto = 0). Si es distinto de cero o `True`, guarda el gráfico en la carpeta `images/figures/`,
            utilizando el título (`title`) como nombre del archivo (en minúsculas y con espacios reemplazados por guiones bajos).
    """

    features = model.feature_names_in_

    cv_scores = cross_val_score(model, X_[features], y_, cv=5, scoring="accuracy")

    df_fold_performance = pd.DataFrame(
        {f"Final{model.__class__.__name__}": cv_scores},
        index=range(1, 6)
    )

    df_fold_performance.plot(figsize=(10, 6), marker="o")

    title = f"Rendimiento en validacion cruzada para {model.__class__.__name__}"
    plt.title(title)
    plt.xlabel("Partición")
    plt.xticks(range(1, 6))
    plt.ylabel("Precisión")
    plt.ylim(0, 1)

    if save:
        plt.savefig(f"images/figures/{model.__class__.__name__}_cross_val_performance.png".lower())

    plt.show()


def plot_model_comparison(model1, model2, X_, y_, save=0):
    """
    Genera un gráfico de barras comparando cuatro métricas de rendimiento (Accuracy, Recall, ROC-AUC y F1-Score)
    entre dos modelos de clasificación distintos.

    :param model1: Primer modelo a comparar. Debe tener ser un DecisionTreeClassifier.

    :param model2: Segundo modelo a comparar. Debe tener ser un RandomForestClassifier.

    :param X_: DataFrame con las variables predictoras.

    :param y_: Variable objetivo.

    :param save: int o bool, opcional (por defecto = 0). Si es distinto de cero o `True`, guarda el gráfico en la ruta indicada por `save_name`.
    """

    metrics = {
        "FinalDecisionTreeClassifier": get_metrics(model1, X_, y_),
        "FinalRandomForestClassifier": get_metrics(model2, X_, y_),
    }

    df_metrics = pd.DataFrame(metrics, index=["Precisión", "Sensibilidad", "ROC-AUC", "Medida F1"])
    df_metrics.plot(kind="bar", figsize=(10, 5))

    plt.title("Comparación de modelos")
    plt.xticks(rotation=0)
    plt.ylabel("Puntuación")

    if save:
        plt.savefig("images/figures/model_comparison.png")

    plt.show()


def plot_learning_curve(model, X_, y_, save=0):
    """
    Genera una curva de aprendizaje que muestra cómo varía el rendimiento (accuracy) del modelo
    con distintos tamaños de muestra de entrenamiento utilizando validación cruzada.

    :param model: Modelo de clasificación que se evaluará. Debe tener `feature_names_in_`.

    :param X_: Dataframe con las variables predictoras.

    :param y_: Variable objetivo.

    :param save: int o bool, opcional. Si es distinto de cero o `True`, guarda el gráfico.
    """

    features = model.feature_names_in_
    train_sizes, train_scores, test_scores = learning_curve(model, X_[features], y_, cv=5, scoring='accuracy')

    title = f"Curva de aprendizaje para {model.__class__.__name__}"
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Entrenamiento", marker='o')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validación", marker='o')
    plt.title(title)
    plt.xlabel("Tamaño del set de entrenamiento")
    plt.ylabel("Precisión")
    plt.legend()

    if save:
        plt.savefig(f"images/figures/{model.__class__.__name__}_learning_curve.png".lower())

    plt.show()


def plot_confusion_matrix(model, X_, y_, save=0):
    """
    Genera y visualiza una matriz de confusión utilizando validación cruzada con 5 particiones.

    :param model: Modelo de clasificación a evaluar. Debe tener `feature_names_in_`.

    :param X_: Variables predictoras.

    :param y_: Variable objetivo.

    :param save: int o bool, opcional. Si es distinto de cero o `True`, guarda el gráfico.
    """

    features = model.feature_names_in_
    y_pred = cross_val_predict(model, X_[features], y_, cv=5)
    conf_matrix = confusion_matrix(y_, y_pred)

    conf_matrix_df = pd.DataFrame(conf_matrix, index=["0 Verdadero", "1 Verdadero"],
                                  columns=["0 Predicho", "1 Predicho"])

    title = f"Matriz de confusion para {model.__class__.__name__}"
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap="Blues", cbar=True, square=True)
    plt.title(title)
    plt.xlabel("Etiquetas predichas")
    plt.ylabel("Etiquetas reales")

    if save:
        plt.savefig(f"images/figures/{model.__class__.__name__}_confusion_matrix.png".lower())

    plt.show()


def plot_scaled_variance(X_, save=0):
    """
    Escala las variables del conjunto de datos mediante Min-Max Scaling y grafica la varianza de cada una de ellas.

    Esto permite visualizar qué variables tienen mayor o menor dispersión relativa después del escalado.

    :param X_: Conjunto de datos con variables numéricas a analizar.

    :param save: int o bool, opcional. Si es distinto de cero o `True`, guarda el gráfico.
    """

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X_)

    variance_df = pd.DataFrame(scaled_features, columns=X_.columns)
    variance = variance_df.var()

    plt.figure(figsize=(14, 6))
    variance.plot(kind="bar", width=0.8)
    plt.title("Varianza por atributo")
    plt.xlabel("Atributo")
    plt.ylabel("Varianza")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save:
        plt.savefig("images/figures/variance.png".lower())

    plt.show()

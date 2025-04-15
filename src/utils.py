import copy

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
    mi_scores_ = mutual_info_classif(X_, y_, random_state=42)
    mi_scores_ = pd.Series(mi_scores_, name="mi_scores", index=X_.columns)
    mi_scores_.sort_values(ascending=False, inplace=True)

    return mi_scores_


def get_metrics(model, X_, y_):
    return [
        cross_val_score(model, X_, y_, cv=5, scoring="accuracy").mean(),
        cross_val_score(model, X_, y_, cv=5, scoring="recall").mean(),
        cross_val_score(model, X_, y_, cv=5, scoring="roc_auc").mean(),
        cross_val_score(model, X_, y_, cv=5, scoring="f1").mean(),
    ]


def get_tuned_model(model, X_, y_, criteria='mean'):

    selector = SelectFromModel(model, threshold=criteria).fit(X_, y_)
    X_select = selector.transform(X_)
    feature_names = X_.columns[selector.get_support()]
    X_select_df = pd.DataFrame(X_select, columns=feature_names)

    model_ = copy.deepcopy(model)
    tuned_model = model_.fit(X_select_df, y_)

    return tuned_model, feature_names


def best_model_per_grid(model, p_grid, X_, y_, criteria='accuracy'):
    grid_search = GridSearchCV(model, p_grid, cv=5, scoring=criteria)
    grid_search.fit(X_, y_)

    print(grid_search.best_params_, grid_search.best_score_)
    return grid_search.best_estimator_


def plot_mi_scores(mi_scores_, save=0, save_name='images/figures/mi_scores.png'):
    mi_scores = mi_scores_[mi_scores_ > 0]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mi_scores.values, y=mi_scores.index, orient='h')
    plt.title('Mutual Information Score')
    plt.xlabel('Score')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save:
        plt.savefig(save_name)

    plt.show()


def plot_distribution(dist, save=0, save_name='images/figures/distribution.png'):
    plt.figure(figsize=(10, 6))
    sns.histplot(dist, kde=True, color='steelblue')
    plt.title("Diagnosis Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of samples")
    plt.xticks(ticks=[0, 1], labels=["Hadn't LCOS", "Had LCOS"])

    if save:
        plt.savefig(save_name)

    plt.show()


def plot_feature_importances(model, title, save=0):
    importances = model.feature_importances_
    feature_names = model.feature_names_in_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.subplots_adjust(left=0.20)
    plt.xlabel('Score')

    if save:
        plt.savefig('images/figures/{}.png'.format(title).lower().replace(' ', '_'))

    plt.show()


def plot_cross_val_performance(model, X_, y_, title, save=0, save_name='images/figures/cross_val.png'):
    cv_scores = cross_val_score(model, X_, y_, cv=5, scoring='accuracy')

    df_fold_performance = pd.DataFrame({"FinalRandomForestClassifier": cv_scores}, index=range(1, 6))

    df_fold_performance.plot(figsize=(10, 6), marker="o")
    plt.title(title)
    plt.xlabel('Fold')
    plt.xticks(range(1, 6))
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    if save:
        plt.savefig(save_name)

    plt.show()


def plot_model_comparison(model1, model2, X_, y_, save=0, save_name='images/figures/model_comparison.png'):
    metrics = {
        'FinalDecisionTreeClassifier': get_metrics(model1, X_, y_),
        'FinalRandomForestClassifier': get_metrics(model2, X_, y_),
    }

    df_metrics = pd.DataFrame(metrics, index=['Accuracy', 'Recall', 'ROC-AUC', 'F1-Score'])
    df_metrics.plot(kind='bar', figsize=(10, 5))
    plt.title('Model Comparison')
    plt.xticks(rotation=0)
    plt.ylabel('Score')

    if save:
        plt.savefig(save_name)

    plt.show()


def plot_learning_curve(model, X_, y_, title, save=0):
    train_sizes, train_scores, test_scores = learning_curve(model, X_, y_, cv=5, scoring='accuracy')
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training', marker='o')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation', marker='o')
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()

    if save:
        plt.savefig('images/figures/{}.png'.format(title).lower().replace(' ', '_'))

    plt.show()


def plot_confusion_matrix(model, X_, y_, title, save=0):
    y_pred = cross_val_predict(model, X_[model.feature_names_in_], y_, cv=5)
    conf_matrix = confusion_matrix(y_, y_pred)

    conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=True, square=True)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    if save:
        plt.savefig('images/figures/{}.png'.format(title).lower().replace(' ', '_'))

    plt.show()


def plot_scaled_variance(X_, save=0, save_name='images/figures/variance.png'):
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X_)

    variance_df = pd.DataFrame(scaled_features, columns=X_.columns)
    variance = variance_df.var()

    plt.figure(figsize=(14, 6))
    variance.plot(kind='bar', width=0.8)
    plt.title('Variance of Each Feature')
    plt.xlabel('Feature')
    plt.ylabel('Variance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save:
        plt.savefig(save_name)

    plt.show()



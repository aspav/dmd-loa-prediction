from __future__ import annotations

import os.path

import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, accuracy_score, f1_score
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score, cumulative_dynamic_auc
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from catboost.core import CatBoostClassifier, CatBoostRegressor, Pool
from sksurv.ensemble import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees
)
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

from sksurv.kernels import clinical_kernel
from sksurv.svm import FastKernelSurvivalSVM

import torch
import torchtuples as tt
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

########################################################################################################################
#   Helper functions for data overview and preprocessing
########################################################################################################################

def missing_values(data):
    # Count number of missing value in a column
    total = data.isnull().sum()           
    
    # Get Percentage of missing values
    percent = (data.isnull().sum()/data.isnull().count()*100)   
    temp = pd.concat([total, percent], axis=1, keys=['Total', 'Percent(%)'])

    # Create a Type column, that indicates the data-type of the column.
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    temp['Types'] = types

    return(np.transpose(temp))


def cleaning_df(data):
    # Iterate over all columns
    for column in data.columns.to_list():
        #check if the column has only NaN values
        if data[column].isnull().sum()==data[column].isnull().count():
            # if it does drop it
            data = data.drop([column], axis=1)
            
    return data


def drop_missing_values(df: pd.DataFrame, percentage: float) -> pd.DataFrame:

    missing_df = missing_values(df) \
        .transpose() \
        .reset_index() \
        .rename(columns={'index': 'feature'}) \
        .sort_values(by='Percent(%)')

    cols_to_drop = missing_df[missing_df['Percent(%)'] >= percentage]['feature'].unique().tolist()
    return df.drop(columns=cols_to_drop)


def preprocess(df, columns, tests): # Create a dataframe with the columns(tests) you want to keep.
    final_df = df[columns] #keep the columns you indicate from the initial dataset
    final_df['tests']=final_df[tests].values.tolist() #create a new column called tests that holds a list of all the other columns values
    final_df = final_df.drop(tests, axis=1)
    final_df = final_df.groupby(['USUBJID', 'VISITNUM']).agg(list)
    return final_df


def make_col(row, col_name: str):
    for entry in row:
        if col_name in entry:
            return entry[1]    
    return np.nan


def join_dfs(df1, df2):
    df2 = df2.join(df1, on=['USUBJID', 'VISITNUM'], how='outer')
    return df2


def make_pairplots(dataframe, var_list):
    ## dataframe: orginal dataframe
    ## var_list: list with the variables we want to plot, the last variable is the class label (hue variable)
    
    sns.set_theme(style="whitegrid", palette='tab10')
    temp_df = dataframe[var_list].reset_index() # we create a df that holds only the variables we want to plot
    sns.pairplot(temp_df, hue=var_list[-1], vars=var_list[:-1]) # we create the pairplot for the variables of interest and with respect to specified class
    plt.show()

########################################################################################################################
#   Imputers
########################################################################################################################


def KNN_imputer(df: pd.DataFrame, columns: List[str]):

    # Takes as input a dataframe -> df and
    # the columns from the specific dataframe that we want to impute using the KNN imputer
    # **Numerical values imputation**

    imputed_cols = KNNImputer().fit_transform(df[columns])

    for i, col in enumerate(columns):
        df[col] = imputed_cols[:, i]
    return df


def simple_imputer(df, columns):

    # Takes as input a dataframe -> df and
    # the columns from the specific dataframe that we want to impute using the Simple imputer
    # **Numerical values imputation**

    imputed_cols = SimpleImputer().fit_transform(df[columns])

    for i, col in enumerate(columns):
        df[col] = imputed_cols[:, i]
    return df


def iterative_imputer(df, columns):

    # Takes as input a dataframe -> df and
    # the columns from the specific dataframe that we want to impute using the Iterative imputer
    # **Numerical values imputation**

    imputed_cols = IterativeImputer().fit_transform(df[columns])

    for i, col in enumerate(columns):
        df[col] = imputed_cols[:, i]
    return df

########################################################################################################################
#   Splitting to train and test
########################################################################################################################


def create_train_test_sets(df, target_columns):

    train_x, test_x, train_y, test_y = train_test_split(
                                        df.drop(target_columns, axis=1),
                                        df[target_columns],
                                        test_size=0.2,
                                        stratify=df['EVENT'],
                                        random_state=0
                                    )

    return train_x, test_x, train_y, test_y


def create_train_test_folds(df, target_columns, n_splits=5):

    return StratifiedKFold(n_splits=n_splits).split(df.drop(target_columns, axis=1), df['EVENT'])

########################################################################################################################
#   Models
########################################################################################################################


def CatBoost_classifier(df, target_columns, train_x, test_x, train_y, test_y):

    cat_columns = df.drop(target_columns, axis=1).select_dtypes(object).columns.to_list()

    _train_y = np.where(train_y[target_columns[0]] == str(True), 1, 0)
    _test_y = np.where(test_y[target_columns[0]] == str(True), 1, 0)

    classifier = CatBoostClassifier(iterations=500, cat_features=cat_columns, verbose=False)

    model = classifier.fit(train_x, _train_y)

    pred = model.predict(test_x)

    return _train_y, _test_y, pred, model


def CatBoost_regressor(df, target_columns, train_x, test_x, train_y, test_y):

    cat_columns = df.drop(target_columns, axis=1).select_dtypes(object).columns.to_list()

    _reg_train_y = train_y[target_columns[1]]
    _reg_test_y = test_y[target_columns[1]]

    regressor = CatBoostRegressor(iterations=500, cat_features=cat_columns, verbose=False)

    reg_model = regressor.fit(train_x, _reg_train_y)

    reg_pred = reg_model.predict(test_x)

    return _reg_train_y, _reg_test_y, reg_pred, reg_model


def ensemble_survival_model(train_x, test_x, train_y, test_y, model_name):

    _train_y = train_y.copy()
    _train_y['EVENT'] = np.where(_train_y['EVENT'] == 'True', True, False)
    _train_y = np.array(
        [(e, t) for e, t in zip(_train_y['EVENT'].to_list(), _train_y['TIME_IN_DAYS'].to_list())],
        dtype=[('Event', 'bool'), ('Time', 'f4')]
    )

    _test_y = test_y.copy()
    _test_y['EVENT'] = np.where(_test_y['EVENT'] == 'True', True, False)
    _test_y = np.array(
        [(e, t) for e, t in zip(_test_y['EVENT'].to_list(), _test_y['TIME_IN_DAYS'].to_list())],
        dtype=[('Event', 'bool'), ('Time', 'f4')]
    )

    if model_name == "coxph":
        model = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, fit_baseline_model=True)
    elif model_name == "rsf":
        model = RandomSurvivalForest()
    elif model_name == "gbsa":
        model = GradientBoostingSurvivalAnalysis()
    elif model_name == "xst":
        model = ExtraSurvivalTrees()
    elif model_name == "c-gbsa":
        model = ComponentwiseGradientBoostingSurvivalAnalysis()
    else:
        raise ValueError(f"Unknown model name '{model_name}'.")

    sa_model = model.fit(train_x, _train_y)
    survival_funcs = sa_model.predict_survival_function(test_x)
    hazard_funcs = sa_model.predict_cumulative_hazard_function(test_x)
    risk_scores = model.predict(test_x)
    shap_values_explainer_model = model.predict

    return survival_funcs, hazard_funcs, risk_scores, sa_model, shap_values_explainer_model


########################################################################################################################
#   Result Reports
########################################################################################################################


def print_report(
        y_event_true: List[np.ndarray],
        y_time_true: List[np.ndarray],
        y_event_predicted: List[np.ndarray],
        y_time_predicted: List[np.ndarray],
        y_event_train: List[np.ndarray] | None = None,
        y_time_train: List[np.ndarray] | None = None,
        risk_scores: List[np.ndarray] | None = None,
        survival_curves: List[np.ndarray] | None = None,
        hazard_curves: List[np.ndarray] | None = None,
        time_range: List[np.ndarray] | None = None,
        tested_model: str | None = None):

    concat_event_true = np.concatenate(y_event_true)
    concat_time_true = np.concatenate(y_time_true)
    concat_event_predicted = np.concatenate(y_event_predicted)
    concat_time_predicted = np.concatenate(y_time_predicted)

    # Micro-averaged report across all splits
    print("\n")
    print("Micro-averaged classification report:")
    print("-------------------------------------")
    print(classification_report(concat_event_true, concat_event_predicted, zero_division=0))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(concat_event_true, concat_event_predicted)).plot()
    plt.show()

    # Micro-averaged MAE
    sample_weights = np.where((concat_event_true == 0) & (concat_event_predicted == 0), 0, 1)
    mae = mean_absolute_error(concat_time_true, concat_time_predicted, sample_weight=sample_weights)
    print(f"Time mean absolute error: {mae:.3f}\n")

    # Micro-averaged Sensitivity against prediction days
    scores = [0]
    days = [0, 90, 180, 270, 360, 540, 720, 900, 1080, 1260]
    for d in days[1:]:
        correct_pred = 0
        correct_true = 0

        for true_event, true_time, predicted_event, predicted_time in \
                zip(concat_event_true, concat_time_true, concat_event_predicted, concat_time_predicted):
            if true_event != 0 and true_time <= d:
                correct_true += 1
                if predicted_event != 0 and predicted_time <= d:
                    correct_pred += 1
        if correct_true == 0:
            scores.append(0)
        else:
            scores.append(correct_pred / correct_true)

    plt.plot(days, scores)
    plt.xlabel("Days")
    plt.ylabel("Sensitivity")
    plt.title("Change in sensitivity as day tolerance increases")
    plt.ylim(0, 1)
    plt.show()
    pd.DataFrame(days, scores).to_csv(f"results/{tested_model}_sensitivity_days.csv")

    scores = [0]
    days = [0, 90, 180, 270, 360, 540, 720, 900, 1080, 1260]
    for d in days[1:]:
        tp, fp, fn = 0, 0, 0

        for true_event, true_time, predicted_event, predicted_time in \
                zip(concat_event_true, concat_time_true, concat_event_predicted, concat_time_predicted):

            if true_event == 1 and true_time <= d:
                if predicted_event == 1 and predicted_time <= d:
                    tp += 1
                elif predicted_event == 0:
                    fn += 1
            elif true_event == 1 and true_time > d or true_event == 0:
                if predicted_event == 1 and predicted_time <= d:
                    fp += 1

        if tp == 0 or tp + fp == 0 or tp + fn == 0:
            scores.append(0)
        else:
            pr = tp / (tp + fp)
            re = tp / (tp + fn)
            f1 = (2 * pr * re) / (pr + re)
            scores.append(f1)

    plt.plot(days, scores)
    plt.xlabel("Days")
    plt.ylabel("F1-score")
    plt.title("Change in F1-score as day tolerance increases")
    plt.ylim(0, 1)
    plt.show()
    pd.DataFrame(days, scores).to_csv(f"results/{tested_model}_f1score_days.csv")

    scores = []
    for true_event, true_time, predicted_event, predicted_time in \
            zip(concat_event_true, concat_time_true, concat_event_predicted, concat_time_predicted):

        if true_event == 1 and predicted_event == 1:
            time_diff = abs(true_time - predicted_time)
            scores.append(time_diff)

    plt.hist(scores, bins=[0, 180, 360, 540, 720, 900, 1080], rwidth=0.5)
    plt.ylabel("Number of predictions")
    plt.xlabel("Days")
    plt.title("Time error in days")
    plt.xticks([180, 360, 540, 720, 900, 1080])
    plt.show()
    pd.DataFrame(scores).to_csv(f"results/{tested_model}_time_diff_days.csv")

    if risk_scores is not None:
        print("C-Index Censored:")
        print("-----------------")

        c_index_sum = 0
        for i, (fold_y_event_true, fold_y_time_true, fold_risk_scores) in \
                enumerate(zip(y_event_true, y_time_true, risk_scores)):

            c_index, n_concordant, n_diconcordant, _, _ = concordance_index_censored(
                fold_y_event_true == 1, fold_y_time_true.tolist(), fold_risk_scores
            )
            c_index_sum += c_index

            print(f"Fold {i}")
            print(f"Concordance Index: {c_index:.3f}")
            print(f"Number of concordant pairs: {n_concordant}")
            print(f"Number of diconcordant pairs: {n_diconcordant}")

        print(f"Macro-average Concordance Index: {c_index_sum / len(risk_scores):.3f}")
        
        count = 0
        if y_event_train is not None and y_time_train is not None:
            print("\n")
            print("C-Index IPCW:")
            print("-------------")

            c_index_ipcw_sum = 0
            for i, (fold_y_event_true, fold_y_time_true, fold_y_event_train, fold_y_time_train, fold_risk_scores) in \
                    enumerate(zip(y_event_true, y_time_true, y_event_train, y_time_train, risk_scores)):

                survival_train = np.array(
                    [(e, t) for e, t in zip(fold_y_event_train.tolist(), fold_y_time_train.to_list())],
                    dtype=[('Event', 'bool'), ('Time', 'f4')]
                )
                survival_test = np.array(
                    [(e, t) for e, t in zip(fold_y_event_true.tolist(), fold_y_time_true.to_list())],
                    dtype=[('Event', 'bool'), ('Time', 'f4')]
                )

                try:
                    c_index_ipcw, n_concordant_ipcw, n_disconco_ipcw, _, _ = concordance_index_ipcw(
                        survival_train,
                        survival_test,
                        fold_risk_scores
                    )
                    c_index_ipcw_sum += c_index_ipcw
                    count += 1
                except:
                    c_index_ipcw, n_concordant_ipcw, n_disconco_ipcw = -1, None, None

                print(f"Fold {i}")
                print(f"IPCW Concordance Index: {c_index_ipcw:.3f}")
                print(f"Number of concordant pairs: {n_concordant_ipcw}")
                print(f"Number of diconcordant pairs: {n_disconco_ipcw}")

            print(f"Macro-average ICPW Concordance Index: {c_index_ipcw_sum / count:.3f}")

    count = 0
    if survival_curves is not None and time_range is not None:
        print("\n")
        print("Integrated Brier Scores:")
        print("-------------")

        brier_score_sum = 0
        for i, (fold_y_event_true, fold_y_time_true, fold_y_event_train, fold_y_time_train, fold_survival_curves, time) in \
                enumerate(zip(y_event_true, y_time_true, y_event_train, y_time_train, survival_curves, time_range)):

            survival_train = np.array(
                [(e, t) for e, t in zip(fold_y_event_train.tolist(), fold_y_time_train.to_list())],
                dtype=[('Event', 'bool'), ('Time', 'f4')]
            )
            survival_test = np.array(
                [(e, t) for e, t in zip(fold_y_event_true.tolist(), fold_y_time_true.to_list())],
                dtype=[('Event', 'bool'), ('Time', 'f4')]
            )

            try:
                score = integrated_brier_score(survival_train, survival_test, fold_survival_curves, time)
                brier_score_sum += score
                count += 1
            except:
                score = None

            print(f"Fold {i}: {score}")

        print(f"Macro-average Integrated Brier Score: {brier_score_sum / count:.3f}")

########################################################################################################################
#   Pipeline
########################################################################################################################

def run_pipeline(df, target_columns, model_name, folds, columns_to_impute=None, threshold_range: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):

    x = df.drop(target_columns, axis=1)
    y = df[target_columns]

    (y_event_true,
     y_time_true,
     y_event_train,
     y_time_train,
     y_event_prediction,
     y_time_predictions,
     risks,
     survival_curves,
     hazard_curves,
     time_range) = [], [], [], [], [], [], [], [], [], []

    list_shap_values = []
    list_test_sets = []

    for i, (train_indices, test_indices) in enumerate(folds):
        print(f"Running fold {i} ...")

        if columns_to_impute is not None:
            x_train = KNN_imputer(x.iloc[train_indices, :], columns=columns_to_impute)
            x_test = KNN_imputer(x.iloc[test_indices, :], columns=columns_to_impute)
        else:
            x_train = x.iloc[train_indices, :]
            x_test = x.iloc[test_indices, :]

        if model_name == "catboost":
            _train_y, _test_y, predictions, cb_clf = (
                CatBoost_classifier(
                    df, target_columns,
                    x_train, x_test,
                    y.iloc[train_indices, :],
                    y.iloc[test_indices, :])
            )

            y_event_true.append(_test_y)
            y_event_prediction.append(predictions)

            _reg_train_y, _reg_test_y, reg_predictions, cb_reg = (
                CatBoost_regressor(
                    df, target_columns,
                    x_train, x_test,
                    y.iloc[train_indices, :],
                    y.iloc[test_indices, :])
            )

            y_time_true.append(_reg_test_y)
            y_time_predictions.append(reg_predictions)

            os.makedirs("results/catboost", exist_ok=True)
            feature_importance = (
                pd.DataFrame({'features': x.columns.to_list(), 'importance': cb_clf.feature_importances_})
                .sort_values('importance', ascending=False)
                .reset_index(drop=True)
            )

            feature_importance.to_csv(f"results/catboost/fold_{i}_feature_importance.csv")

            cat_features = x_test.select_dtypes(object).columns.to_list()
            print(cat_features)

            explainer_clf = shap.TreeExplainer(cb_clf)
            shap_values_clf = explainer_clf.shap_values(Pool(x_test, _test_y, cat_features=cat_features))
            list_shap_values.append(shap_values_clf)
            list_test_sets.append(x_test)
            print("Classifier SHAP values")
            print(shap.summary_plot(shap_values_clf, x_test))
            pd.DataFrame(shap_values_clf, columns=x.columns.to_list()).to_csv(f"results/shap_values/catboost_clf_fold_{i}_feature_shap.csv")
            pd.DataFrame(shap_values_clf).to_csv(f"results/shap_values/catboost_clf_fold_{i}_shap_values.csv")
            pd.DataFrame(x_test).to_csv(f"results/shap_values/catboost_clf_fold_{i}_x_test.csv")

            explainer_reg = shap.TreeExplainer(cb_reg)
            shap_values_reg = explainer_reg.shap_values(Pool(x_test, _reg_test_y, cat_features=cat_features))
            list_shap_values.append(shap_values_reg)
            list_test_sets.append(x_test)
            print("Regressor SHAP values")
            print(shap.summary_plot(shap_values_reg, x_test))
            pd.DataFrame(shap_values_reg, columns=x.columns.to_list()).to_csv(f"results/shap_values/catboost_reg_fold_{i}_feature_shap.csv")
            pd.DataFrame(shap_values_reg).to_csv(f"results/shap_values/catboost_reg_fold_{i}_shap_values.csv")
            pd.DataFrame(x_test).to_csv(f"results/shap_values/catboost_reg_fold_{i}_x_test.csv")

        elif model_name in ["coxph", "rsf", "gbsa", "xst", "c-gbsa"]:

            # Determining classification threshold
            mae_th = []
            acc_th = []
            f1_th = []
            for threshold in threshold_range:
                survival_funcs_th, hazard_funcs_th, risk_scores_th, model, _ = ensemble_survival_model(
                    x_train, x_train,
                    y.iloc[train_indices, :],
                    y.iloc[train_indices, :],
                    model_name
                )
                # Translate to classification predictions
                predictions_th = []
                reg_predictions_th = []
                for j, f in enumerate(survival_funcs_th):
                    survived = True
                    for idx, n in enumerate(f(f.x)):
                        if n < threshold:
                            predictions_th.append(1)
                            reg_predictions_th.append(f.x[idx])
                            survived = False
                            break
                    if survived:
                        predictions_th.append(0)
                        reg_predictions_th.append(f.x[-1])

                sample_weights_th = np.where((np.where(y.iloc[train_indices, 0] == str(True), 1, 0) == 0) & (predictions_th == 0), 0, 1)
                mae_th.append(mean_absolute_error(y.iloc[train_indices, 1], reg_predictions_th, sample_weight=sample_weights_th))
                acc_th.append(accuracy_score(np.where(y.iloc[train_indices, 0] == str(True), 1, 0), predictions_th))
                f1_th.append(accuracy_score(np.where(y.iloc[train_indices, 0] == str(True), 1, 0), predictions_th))

            best_threshold_mae = threshold_range[mae_th.index(min(mae_th))]
            best_threshold_acc = threshold_range[acc_th.index(max(acc_th))]
            best_threshold = (best_threshold_mae + best_threshold_acc)/2
            print(best_threshold)

            survival_funcs, hazard_funcs, risk_scores, m, shap_exp = ensemble_survival_model(
                x_train, x_test,
                y.iloc[train_indices, :],
                y.iloc[test_indices, :],
                model_name
            )

            time = np.arange(
                max(y.iloc[test_indices, 1].min(), min(survival_funcs[0].x)),
                min(y.iloc[test_indices, 1].max(), max(survival_funcs[0].x))
            ).tolist()


            # Translate to classification predictions
            predictions = []
            reg_predictions = []
            for j, f in enumerate(survival_funcs):
                survived = True
                for idx, n in enumerate(f(f.x)):
                    if n < best_threshold:
                        predictions.append(1)
                        reg_predictions.append(f.x[idx])
                        survived = False
                        break
                if survived:
                    predictions.append(0)
                    reg_predictions.append(f.x[-1])
            # For visualization of predictions
                if y.iloc[j, 0] == str(True):
                    plt.step(f.x, f(f.x), c='crimson')
                    plt.axvline(x=y.iloc[j, 1], c='lightblue', linestyle=":")
                else:
                    plt.step(f.x, f(f.x), c='teal')
            plt.ylabel("Survival probability")
            plt.xlabel("Time in days")
            plt.grid(True)
            plt.show()

            y_event_true.append(np.where(y.iloc[test_indices, 0] == str(True), 1, 0))
            y_event_prediction.append(predictions)
            y_event_train.append(np.where(y.iloc[train_indices, 0] == str(True), 1, 0))
            y_time_true.append(y.iloc[test_indices, 1])
            y_time_predictions.append(reg_predictions)
            y_time_train.append(y.iloc[train_indices, 1])

            risks.append(risk_scores)
            survival_curves.append(np.asarray([[f(t) for t in time] for f in survival_funcs]))
            hazard_curves.append(np.asarray([[f(t) for t in time] for f in hazard_funcs]))
            time_range.append(time)

            # Calculating SHAP values
            data = shap.kmeans(x_train, 20)
            explainer = shap.KernelExplainer(shap_exp, data, feature_names=x_train.columns.to_list())
            shap_values = explainer.shap_values(x_test)
            list_shap_values.append(shap_values)
            list_test_sets.append(x_test)
            print(f"{model_name} SHAP values")
            print(shap.summary_plot(shap_values, x_test))
            pd.DataFrame(shap_values, columns=x_test.columns.to_list()).to_csv(f"results/shap_values/{model_name}_fold_{i}_feature_shap.csv")
            pd.DataFrame(shap_values).to_csv(f"results/shap_values/{model_name}_fold_{i}_shap_values.csv")
            pd.DataFrame(x_test).to_csv(f"results/shap_values/{model_name}_fold_{i}_x_test.csv")
        else:
            raise ValueError(f"Unknown model name '{model_name}'.")

    print_report(
        y_event_true,
        y_time_true,
        y_event_prediction,
        y_time_predictions,
        y_event_train if y_event_train else None,
        y_time_train if y_time_train else None,
        risks if risks else None,
        survival_curves if survival_curves else None,
        hazard_curves if hazard_curves else None,
        time_range if time_range else None,
        model_name if model_name else None
    )

    
########################################################################################################################
#   SHAP values calculation and analysis
########################################################################################################################

def shap_values_feature_importance(model, folds):
        model_shap = set([])
        model_shap_0 = set([])
        model_shap_1 = set([])
        model_shap_2 = set([])
        model_shap_3 = set([])
        model_shap_4 = set([])
        model_df = pd.DataFrame()

        for i in range(folds):
            if model=='catboost_clf':
                current_df = pd.read_csv(f'results/shap_values/catboost_clf_fold_{i}_feature_shap.csv').drop(columns='Unnamed: 0')
            elif model=='catboost_reg':
                current_df = pd.read_csv(f'results/shap_values/catboost_reg_fold_{i}_feature_shap.csv').drop(columns='Unnamed: 0')
            else:      
                current_df = pd.read_csv(f'results/shap_values/{model}_fold_{i}_feature_shap.csv').drop(columns='Unnamed: 0')
            vals = np.abs(current_df.values).mean(0)
            shap_importance = pd.DataFrame(list(zip(current_df.columns.to_list(), vals)),
                                          columns=['col_name','feature_importance_vals'])
            shap_importance.sort_values(by=['feature_importance_vals'],
                                       ascending=False, inplace=True)

            model_df[f'fold_{i+1}'] = shap_importance['col_name'].head(10).tolist()

            model_shap.update(shap_importance['col_name'].head(10).tolist())

            if i==0:
                model_shap_0 =set(shap_importance['col_name'].head(10).tolist())
            elif i==1:
                model_shap_1 =set(shap_importance['col_name'].head(10).tolist())
            elif i==2:
                model_shap_2 =set(shap_importance['col_name'].head(10).tolist())
            elif i==3:
                model_shap_3 =set(shap_importance['col_name'].head(10).tolist())
            elif i==4:
                model_shap_4 =set(shap_importance['col_name'].head(10).tolist()) 
            
        summary_df = pd.DataFrame(list(model_shap))
        summary_df[['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']] = np.array(0)
        summary_df = summary_df.rename(columns={0: 'Features'})
        summary_df = summary_df.set_index('Features')

        for fold in summary_df.columns.to_list():
            for feat in summary_df.index.to_list():
                if feat in model_df[fold].values.tolist():
                    summary_df.loc[feat, fold]=1
        summary_df['sum'] = summary_df.sum(axis=1)
        summary_df.sort_values(by=['sum'], ascending=False, inplace=True)                     
            
        return model_shap, model_shap_0, model_shap_1, model_shap_2, model_shap_3, model_shap_4, model_df, summary_df   
    
def make_custom_colormap( cmap_colors ): 
    from colour import Color
    from matplotlib.colors import LinearSegmentedColormap

    color_cmap = LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in cmap_colors ] )
    plt.figure( figsize = (15,3))
    plt.imshow( [list(np.arange(0, len( cmap_colors ) , 0.1)) ] , interpolation='nearest', origin='lower', cmap= color_cmap )
    plt.xticks([])
    plt.yticks([])
    return color_cmap

def concatenate(model, folds):
    
    shap_values_to_concatenate = []
    x_test_to_concatenate = []
    
    for i in range(folds):
            if model=='catboost_clf':
                shap_values_to_concatenate.append(pd.read_csv(f'results/shap_values/catboost_clf_fold_{i}_shap_values.csv')\
                                          .drop(columns='Unnamed: 0'))
                x_test_to_concatenate.append(pd.read_csv(f'results/shap_values/catboost_clf_fold_{i}_x_test.csv')\
                                             .set_index('USUBJID'))
            elif model=='catboost_reg':
                shap_values_to_concatenate.append(pd.read_csv(f'results/shap_values/catboost_reg_fold_{i}_shap_values.csv')\
                                          .drop(columns='Unnamed: 0'))
                x_test_to_concatenate.append(pd.read_csv(f'results/shap_values/catboost_reg_fold_{i}_x_test.csv')\
                                             .set_index('USUBJID'))
            else:      
                shap_values_to_concatenate.append(pd.read_csv(f'results/shap_values/{model}_fold_{i}_feature_shap.csv')\
                                          .drop(columns='Unnamed: 0'))
                x_test_to_concatenate.append(pd.read_csv(f'results/shap_values/{model}_fold_{i}_x_test.csv')\
                                             .set_index('USUBJID'))
                
    shap_values =  np.array(pd.concat(shap_values_to_concatenate))
    x_test = pd.concat(x_test_to_concatenate)
    
    return shap_values, x_test
    
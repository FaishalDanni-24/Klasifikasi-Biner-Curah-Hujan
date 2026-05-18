# Library
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, 
                             confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import seaborn as sns
import os


# Fungsi yang digunakan
def load_dataset(file_path):
    df = pd.read_excel(file_path)

    return df


def prepare_data(df, target_column, selected_features=None):
    if selected_features is not None:
        X = df[selected_features]
    else:
        X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def split_data(X, y, random_state_val, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state_val, stratify=y)
    
    return X_train, X_test, y_train, y_test


def smote_resample(X_train, y_train, using, random_state_val):
    if using is True:
        smote = SMOTE(random_state=random_state_val)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        return X_train_sm, y_train_sm
    
    return X_train, y_train


def build_model(random_state_val, use_randomcv):
    param_dist = {
        'n_estimators': [300, 500, 700],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [5, 10, 15],
        'max_features': ["sqrt", "log2"],
        'bootstrap': [True, False],
        'class_weight':[None, 'balanced']
    }
    model = RandomForestClassifier(random_state=random_state_val, n_jobs=-1)

    if use_randomcv is True:
        model_random = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dist,
            n_iter=100,
            cv = 5, 
            scoring='average_precision',
            random_state=random_state_val,
            n_jobs=-1,
            verbose=0
        )
        return model_random

    return model


def train_model(model, X_train, y_train, use_randomcv):
    model.fit(X_train, y_train)

    if use_randomcv is True:
        print("\n=== Best Estimator from RandomizedSearchCV ===")
        print(model.best_estimator_)

        print("\n=== Best Parameter from RandomizedSearchCV ===")
        print(model.best_params_)
        print

        print("\n=== Best Score From RandomizedSearchCV ===")
        print(model.best_score_)

    return model


def evaluate_model(model, X, y, filepath, dataset_name="Dataset"):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename_suffix = ""

    if dataset_name == "Training Set":
        filename_suffix = "train"
    if dataset_name == "Test Set":
        filename_suffix = "test"

    # Prediksi kelas
    y_pred = model.predict(X)

    # Prediksi probabilitas
    y_scores = model.predict_proba(X)[:, 1]

    # Metrik evaluasi
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # Confusion matrix (TN, FP, FN, TP)
    cm = confusion_matrix(y, y_pred)

    # Kurva ROC
    fpr, tpr, _ = roc_curve(y, y_scores)

    # ROC_AUC
    roc_auc = roc_auc_score(y, y_scores)

    # Kurva PR
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_scores)

    # PR-AUC
    pr_auc = auc(recall_curve, precision_curve)

    # Output hasil
    print(f"\n=== METRIK EVALUASI {dataset_name.upper()} ===")
    print(f"Akurasi  : {accuracy * 100:.2f}%")
    print(f"Presisi  : {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print(f"ROC-AUC  : {roc_auc * 100:.2f}%")
    print(f"PR-AUC   : {pr_auc * 100:.2f}%")

    # Output Confusion Matrix
    print("\nConfusion Matrix:")
    print(cm)
    ax = sns.heatmap(cm, cmap='Blues', annot=True, fmt='d')
    # Berikan label sumbu x dan nilainya. 
    ax.set_xlabel("Predicted", fontsize=14, labelpad=10)
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    # Berikan label sumbu y dan nilainya
    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    # Berikan nama ke plot
    ax.set_title(f"Confusion Matrix for {dataset_name}", fontsize=14, pad=20)
    # Simpan Confusion Matrix
    plt.savefig(f"{filepath}/CM_{filename_suffix}.png")

    # Plot kurva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f'ROC Curve (AUC = {roc_auc:.2f})'
    )
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({dataset_name})')
    plt.legend()
    # Simpan kurva ROC
    plt.savefig(f"{filepath}/ROC-AUC_{filename_suffix}.png")
    # Tampilkan plot
    plt.show()
  
    # Plot kurva PR
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall_curve,
        precision_curve,
        label=f'PR Curve (AUC = {pr_auc:.2f})'
    )
    baseline = sum(y) / len(y)
    plt.axhline(
        y=baseline,
        linestyle='--',
        label=f'Baseline = {baseline:.2f}'
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({dataset_name})')
    plt.legend()
    # Simpan kurva PR
    plt.savefig(f"{filepath}/PR-AUC_{filename_suffix}.png")
    # Tampilkan plot
    plt.show()


def show_feature_importance(model, feature_names):
    print("\n=== Feature Importance ===")

    importances = model.feature_importances_

    for feature, imp in zip(feature_names, importances):
        print(feature, ":", round(imp, 3))
    

def save_model(model, filepath):
    pickle.dump(model, open(f"{filepath}/RF_model.pkl", "wb"))
    print("Best Random Forest Model is saved!")

# Program utama
def main():
    # Load dataset
    df = load_dataset("data-cuaca-serang_klasifikasi-biner.xlsx")

    # Fitur dan Target yang dipilih
    features_all = ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "FF_AVG", 
                    "DDD_CAR_C", "DDD_CAR_E", "DDD_CAR_N", "DDD_CAR_NE",
                    "DDD_CAR_NW", "DDD_CAR_S", "DDD_CAR_SE", "DDD_CAR_SW", 
                    "DDD_CAR_W", "SIN_DDD_X", "COS_DDD_X"]
    features = ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "SIN_DDD_X", "COS_DDD_X"]
    target = 'C_RR'

    # Konfigurasi model
    use_SMOTE = True
    use_RANDOMSEARCHCV = True
    # Nilai Seed untuk replikasi
    random_state_set = 50

    print("Random state seed:", random_state_set)
    print("Using SMOTE:", use_SMOTE)
    print("Using Algorithm: Random Forest")
    print("Using RANDOMSEARCHCV:", use_RANDOMSEARCHCV)

    # Pemisahan data menjadi fitur dan target
    X, y = prepare_data(df, target_column=target, selected_features=features)

    # Splitting data menjadi set training dan set test
    X_train, X_test, y_train, y_test = split_data(X, y, random_state_val=random_state_set, test_size=0.2)

    # Menerapkan SMOTE set training
    X_train_sm, y_train_sm = smote_resample(X_train, y_train, use_SMOTE, random_state_set)

    # Buat model
    rain_classifier_model = build_model(random_state_set, use_RANDOMSEARCHCV)

    # Train model
    rain_classifier_model = train_model(rain_classifier_model, X_train_sm, y_train_sm, use_RANDOMSEARCHCV)

    # Pilih model terbaik dari RandomizedSearchCV
    if use_RANDOMSEARCHCV:
        best_rain_classifier_model = rain_classifier_model.best_estimator_
    else:
        best_rain_classifier_model = rain_classifier_model

    # Feature importance
    show_feature_importance(best_rain_classifier_model, features)

    # Evaluasi model pada set training
    evaluate_model(best_rain_classifier_model, X_train_sm, y_train_sm, "figures/Model/RF", "Training Set")

    # Evaluasi model pada set test
    evaluate_model(best_rain_classifier_model, X_test, y_test, "figures/Model/RF", "Test Set")

    # Simpan model
    save_model(best_rain_classifier_model, "models")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, cv2, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

# =============================
# CONFIGURATION
# =============================
RANDOM_SEED = 42
IMG_SIZE = 224
DATASET_DIR = "cashew label dataset"
RESULTS_DIR = "cashew result"
N_FOLDS = 5
N_TRIALS = 50   # Optuna trials per model

# =============================
# Create result folders
# =============================
def make_dirs(base=RESULTS_DIR):
    dirs = {
        "reports": os.path.join(base, "reports"),
        "confusion": os.path.join(base, "confusion_matrices"),
        "visual": os.path.join(base, "visualizations"),
        "summary": os.path.join(base, "summary")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

# =============================
# Feature extraction
# =============================
def extract_features_one(img_bgr):
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    col_mean = hsv.reshape(-1,3).mean(axis=0)
    col_std  = hsv.reshape(-1,3).std(axis=0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Texture (GLCM)
    glcm = graycomatrix(gray, [5], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    energy   = graycoprops(glcm, 'energy')[0,0]
    homo     = graycoprops(glcm, 'homogeneity')[0,0]
    corr     = graycoprops(glcm, 'correlation')[0,0]

    # Shape (Aspect Ratio)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect = w/(h+1e-6)
    else:
        aspect = 1.0

    # HOG
    hog_feats = hog(gray, orientations=9, pixels_per_cell=(16,16),
                    cells_per_block=(2,2), block_norm='L2-Hys', visualize=False)
    hog_feats = np.mean(hog_feats.reshape(-1,9), axis=0)

    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0,10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    # Hu Moments
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()

    return np.hstack([col_mean, col_std, contrast, energy, homo, corr,
                      aspect, hog_feats, lbp_hist, hu])

def build_feature_matrix(df):
    feats, y = [], []
    for fp, lbl in tqdm.tqdm(zip(df['filepath'], df['label_idx']), total=len(df)):
        im = cv2.imread(fp)
        if im is None: continue
        feats.append(extract_features_one(im))
        y.append(lbl)
    return np.vstack(feats).astype(np.float32), np.array(y)

# =============================
# Confusion Matrix Plot
# =============================
def plot_confusion_matrix(y_true, y_pred, classes, outpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# =============================
# Optuna Hyperparameter Tuning
# =============================
def objective(trial, model_name, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    if model_name == "SVM":
        C = trial.suggest_loguniform("C", 1e-2, 1e2)
        gamma = trial.suggest_loguniform("gamma", 1e-4, 1e-1)
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        clf = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=RANDOM_SEED)

    elif model_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 30])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     max_features=max_features, random_state=RANDOM_SEED)

    elif model_name == "MLP":
        hidden1 = trial.suggest_int("hidden1", 128, 512)
        hidden2 = trial.suggest_int("hidden2", 64, 256)
        hidden3 = trial.suggest_int("hidden3", 32, 128)
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        learning_rate_init = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        clf = MLPClassifier(hidden_layer_sizes=(hidden1, hidden2, hidden3),
                            activation=activation, solver="adam",
                            learning_rate_init=learning_rate_init,
                            max_iter=500, random_state=RANDOM_SEED, early_stopping=True)

    else:
        raise ValueError("Unsupported model")

    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    return np.mean(scores)

def tune_with_optuna(model_name, X, y):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_name, X, y), n_trials=N_TRIALS)
    print(f"Best {model_name} params:", study.best_params)
    return study.best_params

# =============================
# Main
# =============================
def main():
    dirs = make_dirs()

    # Load dataset
    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    filepaths, labels = [], []
    for c in classes:
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif"):
            for fp in glob.glob(os.path.join(DATASET_DIR, c, ext)):
                filepaths.append(fp); labels.append(c)
    df = pd.DataFrame({"filepath": filepaths, "label": labels})
    df['label_idx'] = df['label'].astype('category').cat.codes

    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=RANDOM_SEED)
    X_train, y_train = build_feature_matrix(df_train)
    X_test, y_test   = build_feature_matrix(df_test)

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Tune models
    best_params_svm = tune_with_optuna("SVM", X_train, y_train)
    best_params_rf  = tune_with_optuna("RandomForest", X_train, y_train)
    best_params_mlp = tune_with_optuna("MLP", X_train, y_train)

    # Train final models
    svm = SVC(**best_params_svm, probability=True, random_state=RANDOM_SEED)
    rf = RandomForestClassifier(**best_params_rf, random_state=RANDOM_SEED)
    mlp = MLPClassifier(hidden_layer_sizes=(best_params_mlp["hidden1"], best_params_mlp["hidden2"], best_params_mlp["hidden3"]),
                        activation=best_params_mlp["activation"],
                        learning_rate_init=best_params_mlp["lr"],
                        max_iter=500, random_state=RANDOM_SEED, early_stopping=True)

    # Ensemble
    ensemble = VotingClassifier(estimators=[("svm", svm), ("rf", rf), ("mlp", mlp)], voting="soft")

    models = {"SVM": svm, "RandomForest": rf, "MLP": mlp, "Ensemble": ensemble,
              "KNN": KNeighborsClassifier(n_neighbors=7),
              "NaiveBayes": GaussianNB()}

    all_results = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec  = recall_score(y_test, preds, average="macro", zero_division=0)
        f1   = f1_score(y_test, preds, average="macro", zero_division=0)

        all_results.append([name, acc, prec, rec, f1])

        rep = classification_report(y_test, preds, target_names=classes, zero_division=0)
        with open(os.path.join(dirs["reports"], f"{name}_report.txt"), "w") as f:
            f.write(rep)

        cm_path = os.path.join(dirs["confusion"], f"{name}_cm.png")
        plot_confusion_matrix(y_test, preds, classes, cm_path)

    # Save summary
    df_summary = pd.DataFrame(all_results, columns=["Model","Accuracy","Precision","Recall","F1"])
    df_summary.to_csv(os.path.join(dirs["summary"], "metrics_summary.csv"), index=False)

if __name__ == "__main__":
    main()

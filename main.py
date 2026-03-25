import argparse
import os
import pickle
import json
from matplotlib import pyplot as plt
import numpy as np
import scipy
import math
import pandas
from time import gmtime, strftime
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, auc, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# exemple pour linear regression: 
# python3 main.py --dataset_path="df_clean.csv" --ml_method="log_reg" --save_dir="results" --action="model_evaluation"

# Arguments
parser = argparse.ArgumentParser(
        prog = "25/26 ML Project Example",
        description = "Example program for the ML Project course (2025/2026 M1 IDD)")

parser.add_argument("--dataset_path", type = str, default = "", help = "path to the dataset file")
parser.add_argument("--ml_method", type = str, default = "KNN", help = "name of the ML method to use ('KNN', 'GNB', 'log_reg')")

parser.add_argument("--action", type = str, default = "", help = "the current task: model evaluation or hyper-parameter(s) tuning")

parser.add_argument("--knn_n_neighbors", type = int, default = 5, help = "number of nearest neighbors when using KNN")

parser.add_argument("--cv_nsplits", type = int, default = 5, help = "cross-validation: number of splits")

parser.add_argument("--save_dir", type = str, default = "", help = "where to save the model, the logs and the configuration")

args = parser.parse_args()

# Create the directory containing the model, the logs, etc.
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir)

path_model = os.path.join(out_dir, "model.pkl")
path_config = os.path.join(out_dir, "config.json")
path_logs = os.path.join(out_dir, "logs.json")
plots_dir = os.path.join(out_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Store the configuration
with open(path_config, "w") as f:
    json.dump(vars(args), f)

# Loading the dataset
df = pandas.read_csv(args.dataset_path)

# Build the features and the targets
# usagers
features_usagers = ['place', 'catu', 'sexe', 'age', 
       'aucun', 'ceinture', 'casque', 'dispositif_enfant', 'non determinable', 
       'gilet_reflechissant', 'airbag', 'gants', 'autre']

binary_features_usagers = ['aucun', 'ceinture', 'casque', 'dispositif_enfant', 'gilet_reflechissant', 'airbag', 'gants', 'autre', 'non determinable']
numerical_features_usagers = ["age"]
categorical_features_usagers = list(set(features_usagers) - set(binary_features_usagers) - set(numerical_features_usagers))

# lieux
categorical_features_lieux = ['catr', 'circ', 'vosp', 'prof', 'plan', 'surf','infra', 'situ']
numerical_features_lieux = ["vma", 'nbv']

features_lieux = categorical_features_lieux + numerical_features_lieux

# vehicules
categorical_features_vehicules = ['catv', 'obs', 'obsm', 'choc', 'manv', 'motor',"senc"]
numerical_features_vehicules = []
features_vehicules = categorical_features_vehicules + numerical_features_vehicules

# caract
categorical_features_caract = ['lum', 'agg', 'int', 'atm', 'col']
binary_features_caract = ["is_weekend"]     
numerical_features_caract = ['lat', 'long', 'day_sin',
       'day_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
       'hour_sin', 'hour_cos']

features_caract = categorical_features_caract + binary_features_caract + numerical_features_caract

num_features = numerical_features_usagers + numerical_features_caract
cat_features = categorical_features_usagers + categorical_features_caract + categorical_features_lieux + categorical_features_vehicules
bin_features = binary_features_usagers + binary_features_caract

features = num_features + cat_features + bin_features
target = "grav_bin"

df.dropna(subset=features + [target], inplace = True)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build the model
if args.ml_method == "KNN":
    model = KNeighborsClassifier(n_neighbors = args.knn_n_neighbors)
elif args.ml_method == "GNB":
    model = GaussianNB()
elif args.ml_method == "RandomForest":
    model = RandomForestClassifier(random_state=42)
elif args.ml_method == "GradientBoosting":
    model = GradientBoostingClassifier(random_state=42)
elif args.ml_method == "log_reg":
    model = LogisticRegression(max_iter = 1000)
else:
    raise ValueError(f"Invalid value found for argument 'ml_method': found '{args.ml_method}'")

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("bin", "passthrough", bin_features)
    ]
)

# Build the pipeline
pipeline = Pipeline([("preprocess", preprocessor),("model", model)])

if args.action == "model_evaluation":
    pass
elif args.action == "hyper_parameter_tuning":
    param_grid = {
        "model__n_neighbors": [i for i in range(5, 21, 2)],   # ex: 5 à 20
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=args.cv_nsplits,
        n_jobs=-1,
        verbose=2
    )

    # Fitting GridSearchCV
    grid.fit(X_train, y_train)

    # Meilleur pipeline trouvé
    pipeline = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV F1:", grid.best_score_)

else:
    raise ValueError(f"Invalid value found for argument 'action': found '{args.action}'")

# Fit the data
pipeline.fit(X_train, y_train)

# Evaluation of the model
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

report = classification_report(y_test, y_pred, output_dict=True)

with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump({"summary": metrics, "classification_report": report}, f, indent=4)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
plt.close()

# ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
plt.close()

# Save model
with open(path_model, 'wb') as f:
    pickle.dump({"pipeline": pipeline}, f)

# Test model - In our case : f1
lst_scores = np.array(cross_val_score(pipeline, X, y, cv = args.cv_nsplits, scoring="f1"))
score = sum(lst_scores) / args.cv_nsplits

# Store results
with open(path_logs, "w") as f:
    json.dump({"score": score}, f)


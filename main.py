import argparse
import os
import pickle
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas
from time import gmtime, strftime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, auc, classification_report, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# python3 main.py model_training --dataset_path="df_clean.csv" --save_dir="results" LogisticRegression

# python3 main.py hyper_parameter_tuning --dataset_path="df_clean.csv" --save_dir="results" --sample="undersampling" LogisticRegression
# python3 main.py hyper_parameter_tuning --dataset_path="df_clean.csv" --save_dir="results" --sample="undersampling" RandomForest
# python3 main.py hyper_parameter_tuning --dataset_path="df_clean.csv" --save_dir="results" --sample="undersampling" GradientBoosting

# ONLY with THE BEST model: test on test set
# python3 main.py model_evaluation --dataset_path="df_clean.csv" --save_dir="results" --load_model_path="models/RandomForest-hpt-os-2026-03-27_17-03-59/model.pkl"

# Arguments 
parser = argparse.ArgumentParser(
    prog="25/26 ML Project Group: Nerdjes Ahdad and Ekaterina Galkina",
    description="ML Project (2025/2026 M1 IDD): road accident severity prediciton"
)

subparsers = parser.add_subparsers(dest="action", required=True)

# Common parent parser
common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file")
common_parser.add_argument("--save_dir", type=str, required=True, help="Output directory")

# Train parser
train_parser = subparsers.add_parser("model_training", parents=[common_parser],
                                     help="Train a model")
train_parser.add_argument("--sample", type=str, default="passthrough",
                          choices=["passthrough", "undersampling", "oversampling"],
                          help="Sampling strategy to handle class imbalance")

# Subparsers for each ML method - "LogisticRegression", "RandomForest", "GradientBoosting"
train_subparsers = train_parser.add_subparsers(dest="ml_method", required=True, help="ML model to use")

# Logistic Regression
logreg_parser = train_subparsers.add_parser("LogisticRegression", help="Train Logistic Regression")
logreg_parser.add_argument("--C", type=float, nargs=None, default=0.1, choices=[0.01, 0.1, 1, 10, 100])
logreg_parser.add_argument("--penalty", type=str, nargs=None, default="l2", choices=["l1", "l2"])
logreg_parser.add_argument("--class_weight", type=str, nargs=None, default=None, choices=[None, "balanced"])

# Random Forest
rf_parser = train_subparsers.add_parser("RandomForest", help="Train Random Forest")
rf_parser.add_argument("--n_estimators", type=int, nargs=None, default=100, choices=[100, 200])
rf_parser.add_argument("--max_depth", type=int, nargs=None, default=None, choices = [None, 10])
rf_parser.add_argument("--class_weight", type=str, nargs=None, default=None, choices=[None, "balanced"])

# Gradient Boosting
gb_parser = train_subparsers.add_parser("GradientBoosting", help="Train Gradient Boosting")
gb_parser.add_argument("--n_estimators", type=int, nargs=None, default=100, choices=[100, 200])
gb_parser.add_argument("--learning_rate", type=float, nargs=None, default=0.01, choices=[0.01, 0.1])

# Tune
tune_parser = subparsers.add_parser("hyper_parameter_tuning", parents=[common_parser], help="Tune hyperparameters")
tune_parser.add_argument("--sample", type=str, default="passthrough", choices=["passthrough", "undersampling", "oversampling"], help="Sampling strategy to handle class imbalance")
tune_parser.add_argument("--cv_nsplits", type=int, default=5, help="Number of CV folds")
tune_subparsers = tune_parser.add_subparsers(dest="ml_method", required=True, help="ML model to use")
tune_subparsers.add_parser("LogisticRegression", help="Tune Logistic Regression")
tune_subparsers.add_parser("RandomForest", help="Tune Random Forest")
tune_subparsers.add_parser("GradientBoosting", help="Tune Gradient Boosting")

# Evaluate
eval_parser = subparsers.add_parser("model_evaluation", parents=[common_parser], help="Evaluate a saved model")
eval_parser.add_argument("--load_model_path", type=str, required=True, help="Path to saved model.pkl")

args = parser.parse_args()

# Create the directory containing the model, the logs, etc.
time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

act = "me" if args.action == "model_evaluation" else "hpt" if args.action == "hyper_parameter_tuning" else "mt" # model_testing

if args.action == "model_evaluation":
    model_folder = os.path.basename(os.path.dirname(args.load_model_path))
    txt_prefix = "-".join([model_folder.split("-")[0], model_folder.split("-")[2]])
    dir_name = f"{txt_prefix}-{act}-{time}"
else:
    samp = "us" if args.sample == "undersampling" else "os" if args.sample == "oversampling" else "pass"
    dir_name = f"{args.ml_method}-{act}-{samp}-{time}"

out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir)

path_model = os.path.join(out_dir, "model.pkl")
path_config = os.path.join(out_dir, "config.json")
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
numerical_features_vehicules =  []
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

X = df[features]
y = df[target]

# stratify -> to ensure the proportion of target labels stays the same in splits, as our data is unbalanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)

# Build the model - when we are evaluating a model we are not training it so we necessarly retrieve an existing one
if args.action != "model_evaluation":
    MODELS = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="saga", random_state=20),
        "RandomForest": RandomForestClassifier(random_state=20, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=20)
    }

    if args.ml_method not in MODELS:
        raise ValueError(f"Invalid ml_method: {args.ml_method}")

    model = MODELS[args.ml_method]

    # Preprocessing
    transformer_list = []
    if num_features: # On n'ajoute que si la liste n'est pas vide
        transformer_list.append(("num", StandardScaler(), num_features))

    if cat_features:
        transformer_list.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))

    if bin_features:
        transformer_list.append(("bin", "passthrough", bin_features))

    preprocessor = ColumnTransformer(transformers=transformer_list)

    sampler = "passthrough"
    if args.sample == "undersampling":
        sampler = RandomUnderSampler(random_state=42)
    elif args.sample == "oversampling":
        sampler = RandomOverSampler(random_state=42)

    # Build the pipeline
    pipeline = ImbPipeline([("preprocess", preprocessor),("sample", sampler),("model", model)])

if args.action == "model_training":
    # Fit the data simply 
    pipeline.fit(X_train, y_train)

elif args.action == "hyper_parameter_tuning":
    if args.ml_method == "LogisticRegression":
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"]
        }
    elif args.ml_method == "RandomForest":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, None],
            "model__class_weight": [None, "balanced"]
        }
    elif args.ml_method == "GradientBoosting":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1]
        }

    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=StratifiedKFold(n_splits=args.cv_nsplits, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2
    )

    # Fitting GridSearchCV
    grid.fit(X_train, y_train)

    path_best_parameters = os.path.join(out_dir, "best_parameters.json")

    # Meilleur pipeline trouve
    pipeline = grid.best_estimator_
    with open(path_best_parameters, "w") as f:
        json.dump({"best_parameters": grid.best_params_, "best_cv_f1": grid.best_score_}, f, indent=4)

elif args.action == "model_evaluation": # Here we necessarly load a pre-trained model
    if args.load_model_path:
        with open(args.load_model_path, "rb") as f:
            pipeline = pickle.load(f)["pipeline"]
        print(f"Loaded model from {args.load_model_path}")
    else:
        raise ValueError(f"Invalid value found for argument 'model_evaluation': found '{args.load_model_path}'")

else:
    raise ValueError(f"Invalid value found for argument 'action': found '{args.action}'")

def recover_original_feature(feature_name):
    # numerical features: keep full original name
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "", 1)

    # binary features: keep full original name
    elif feature_name.startswith("bin__"):
        return feature_name.replace("bin__", "", 1)

    # categorical one-hot encoded features: recover original categorical column
    elif feature_name.startswith("cat__"):
        cat_part = feature_name.replace("cat__", "", 1)

        for col in cat_features:
            if cat_part == col or cat_part.startswith(col + "_"):
                return col

        return cat_part  # fallback if no match found

    return feature_name

# analyser les coefficients selon le model and we are not evaluating the final model
if args.action != "model_evaluation" and args.ml_method == "LogisticRegression": 
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
    coefficients = pipeline.named_steps['model'].coef_[0]  # contribution to class 1

    feature_importance = pandas.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients)
    })

    # Detailed coefficients
    feature_importance.sort_values(by="abs_coefficient", ascending=False).to_csv(
        os.path.join(out_dir, "feature_importance_detailed_logreg.csv"),
        index=False
    )

    # Aggregated coefficients by original feature
    feature_importance["original_feature"] = feature_importance["feature"].apply(recover_original_feature)

    aggregated_importance = (
        feature_importance
        .groupby("original_feature", as_index=False)["abs_coefficient"]
        .sum()
        .sort_values(by="abs_coefficient", ascending=False)
    )

    aggregated_importance.to_csv(
        os.path.join(out_dir, "feature_importance_aggregated_logreg.csv"),
        index=False
    )

if args.action != "model_evaluation" and args.ml_method in ["RandomForest", "GradientBoosting"]:
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    feature_importance = pandas.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Detailed importances (after one-hot encoding)
    feature_importance.sort_values(by="importance", ascending=False).to_csv(
        os.path.join(out_dir, f"feature_importance_detailed_{args.ml_method}.csv"),
        index=False
    )

    # Aggregated importances by original feature
    feature_importance["original_feature"] = feature_importance["feature"].apply(recover_original_feature)

    aggregated_importance = (
        feature_importance
        .groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values(by="importance", ascending=False)
    )

    aggregated_importance.to_csv(
        os.path.join(out_dir, f"feature_importance_aggregated_{args.ml_method}.csv"),
        index=False
    )


# When we are selecting a model we are not looking at the final score on the test set
if args.action == "model_evaluation":
    # Evaluation of the model
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"classification_report": report}, f, indent=4)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    plt.title("Confusion Matrix on the test set")
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
    plt.title("ROC Curve on the test set")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.close()
else:
    # Evaluation of the model on the train set - we don't touch yet the test split
    y_pred = pipeline.predict(X_train)
    y_proba = pipeline.predict_proba(X_train)[:, 1]

    report = classification_report(y_train, y_pred, output_dict=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"classification_report on the train set": report}, f, indent=4)

    # Confusion Matrix
    cm = confusion_matrix(y_train, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    plt.title("Confusion Matrix on the train set")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_train, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on the train set")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.close()

    # Save model
    with open(path_model, 'wb') as f:
        pickle.dump({"pipeline": pipeline}, f)

# Sound notification once finished
os.system('afplay /System/Library/Sounds/Glass.aiff')
from pathlib import Path

import pandas as pd



def load_data(file_name: str = "dataset-uci.csv") -> pd.DataFrame:
    data_path = Path(file_name)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    return pd.read_csv(data_path)

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split  # type: ignore[reportMissingModuleSource]

    X = df.drop(columns=["Gallstone Status", "Hepatic Fat Accumulation (HFA)"])
    Y = df["Gallstone Status"].map({0: 1, 1: 0})

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=Y
    )

    return X_train, X_test, Y_train, Y_test

def train_baseline(X_train, Y_train):
    from sklearn.dummy import DummyClassifier  # type: ignore[reportMissingModuleSource]

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, Y_train)
    return dummy_clf

def evaluate_baseline(X_train, Y_train):
    from sklearn.dummy import DummyClassifier  # type: ignore[reportMissingModuleSource]
    from sklearn.model_selection import cross_val_score, StratifiedKFold  # type: ignore[reportMissingModuleSource]

    dummy_clf = DummyClassifier(strategy="most_frequent")
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(dummy_clf, X_train, Y_train, cv=skfold, scoring="accuracy")

    return scores

def build_preprocessor(binary_cols, categorical_cols, numeric_cols):
    from sklearn.compose import ColumnTransformer  # type: ignore[reportMissingModuleSource]
    from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore[reportMissingModuleSource]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("binary", "passthrough", binary_cols),
            ("categorical", OneHotEncoder(handle_unknown = "ignore"), categorical_cols)
        ]
    )
    return preprocessor

def build_pipeline(preprocessor):
    from sklearn.pipeline import Pipeline # type: ignore[reportMissingModuleSource]
    from sklearn.linear_model import LogisticRegression # type: ignore[reportMissingModuleSource]
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter = 1000))
    ])

    return pipeline

def eval_pipeline(X_train, Y_train, binary_cols, categorical_cols, numerical_cols):
    from sklearn.model_selection import cross_validate, StratifiedKFold  # type: ignore[reportMissingModuleSource]
    preprocessor = build_preprocessor(binary_cols, categorical_cols, numerical_cols)
    pipeline = build_pipeline(preprocessor)
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X_train, Y_train, cv=skfold, 
                            scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall"})
    return scores

def get_oof_probabilities(pipeline, X_train, Y_train, cv, method):
    from sklearn.model_selection import cross_val_predict # type: ignore[reportMissingModuleSource]
    return cross_val_predict(pipeline, X_train, Y_train, cv=cv, method=method)

if __name__ == "__main__":
    df = load_data()
    if df.empty:
        print("DataFrame is empty.")
    else:
        binary_cols = ["Gender", 
                       "Coronary Artery Disease (CAD)", 
                       "Hypothyroidism", 
                       "Hyperlipidemia", 
                       "Diabetes Mellitus (DM)"]
        categorical_cols = ["Comorbidity"]
        numeric_cols = ["Age", 
                        "Height", 
                        "Weight", 
                        "Body Mass Index (BMI)", 
                        "Total Body Water (TBW)", 
                        "Extracellular Water (ECW)", 
                        "Intracellular Water (ICW)", 
                        "Extracellular Fluid/Total Body Water (ECF/TBW)", 
                        "Total Body Fat Ratio (TBFR) (%)", 
                        "Lean Mass (LM) (%)", 
                        "Body Protein Content (Protein) (%)",
                        "Visceral Fat Rating (VFR)", 
                        "Bone Mass (BM)", 
                        "Muscle Mass (MM)", 
                        "Obesity (%)", 
                        "Total Fat Content (TFC)", 
                        "Visceral Fat Area (VFA)", 
                        "Visceral Muscle Area (VMA) (Kg)", 
                        "Glucose", 
                        "Total Cholesterol (TC)", 
                        "Low Density Lipoprotein (LDL)", 
                        "High Density Lipoprotein (HDL)", 
                        "Triglyceride", 
                        "Aspartat Aminotransferaz (AST)", 
                        "Alanin Aminotransferaz (ALT)", 
                        "Alkaline Phosphatase (ALP)", 
                        "Creatinine", 
                        "Glomerular Filtration Rate (GFR)", 
                        "C-Reactive Protein (CRP)",
                        "Hemoglobin (HGB)", 
                        "Vitamin D"]

        
        from sklearn.model_selection import StratifiedKFold # type: ignore[reportMissingModuleSource]
        from sklearn.metrics import precision_score, recall_score, confusion_matrix # type: ignore[reportMissingModuleSource]
        X_train, X_test, Y_train, Y_test = split_data(df)
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        scores = evaluate_baseline(X_train, Y_train)
        print(f"Baseline CV accuracy: {scores.mean():.2%}")

        pipeline_scores = eval_pipeline(
            X_train, Y_train, binary_cols, categorical_cols, numeric_cols
        )
        for metric in ("accuracy", "precision", "recall"):
            print(
                f"Pipeline CV {metric}: "
                f"{pipeline_scores[f'test_{metric}'].mean():.2%}"
            )

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        preprocessor = build_preprocessor(binary_cols, categorical_cols, numeric_cols)
        pipeline = build_pipeline(preprocessor)

        probabilities = get_oof_probabilities(
            pipeline, X_train, Y_train,
            cv=folds,
            method="predict_proba",
        )
        disease_probs = probabilities[:, 1]
        for threshold in (0.3, 0.5):
            print(f"Threshold: {threshold}")
            predictions = (disease_probs >= threshold).astype(int)
            precision = precision_score(Y_train, predictions)
            recall = recall_score(Y_train, predictions)
            cm = confusion_matrix(Y_train, predictions, labels=[0, 1])

            # Temporary check while implementing out-of-fold predictions.
            print(
                f"Disease probabilities: shape={disease_probs.shape}, "
                f"min={disease_probs.min()}, max={disease_probs.max()}"
            )
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            print("Confusion Matrix:")
            print(cm)
            
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

# Configuration
DATASET_PATH = r"C:\Codesoft\DSA_3_T5\creditcard.csv"
MODEL_DIR = r"C:\Users\DELL\.gemini\antigravity\scratch\credit_card_fraud_detection"
MODEL_FILE = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")


def load_and_preprocess_data():
    print(f"\n[1/5] Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please check the path.")

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded. Shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")

    # Scale 'Time' and 'Amount' features (V1-V28 are already PCA scaled features)
    print("\n[2/5] Normalizing 'Time' and 'Amount' features...")
    scaler = RobustScaler()  # Robust to outliers (common in transaction amounts)
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Drop original unscaled features
    df = df.drop(['Time', 'Amount'], axis=1)

    # Move Class to the end to keep features organized
    class_col = df['Class']
    df = df.drop(['Class'], axis=1)
    df['Class'] = class_col

    # Feature-target split
    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y, scaler


def handle_imbalance(X_train, y_train, method="balanced"):
    """
    Handles class imbalance using three possible methods:
    1. 'balanced': Do nothing to data, use class_weight='balanced' in models.
    2. 'undersample': Undersample the majority class to match the minority class.
    3. 'oversample': Oversample the minority class to match the majority class.
    """
    print(f"\n[3/5] Handling class imbalance using method: '{method}'...")

    # Combine training data for resampling
    train_data = pd.concat([X_train, y_train], axis=1)
    genuine = train_data[train_data.Class == 0]
    fraud = train_data[train_data.Class == 1]

    print(f"Original training class sizes -> Genuine: {len(genuine)}, Fraud: {len(fraud)}")

    if method == "undersample":
        # Random undersample majority class to match minority class
        genuine_undersampled = genuine.sample(n=len(fraud), random_state=42)
        balanced_train = pd.concat([genuine_undersampled, fraud])
        X_resampled = balanced_train.drop('Class', axis=1)
        y_resampled = balanced_train['Class']
        print(f"Resampled training class sizes -> Genuine: {len(genuine_undersampled)}, Fraud: {len(fraud)}")
        return X_resampled, y_resampled

    elif method == "oversample":
        # Random oversample minority class to match majority class
        fraud_oversampled = fraud.sample(n=len(genuine), replace=True, random_state=42)
        balanced_train = pd.concat([genuine, fraud_oversampled])
        X_resampled = balanced_train.drop('Class', axis=1)
        y_resampled = balanced_train['Class']
        print(f"Resampled training class sizes -> Genuine: {len(genuine)}, Fraud: {len(fraud_oversampled)}")
        return X_resampled, y_resampled

    else:
        # 'balanced' class weights method (no changes to the dataset directly)
        print("Using model class weighting. No data resampling applied.")
        return X_train, y_train


def train_model(X_train, y_train, algorithm="logistic_regression", use_class_weights=False):
    print(f"\n[4/5] Training {algorithm.upper()} model...")

    class_weight = "balanced" if use_class_weights else None

    if algorithm == "logistic_regression":
        model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
    elif algorithm == "random_forest":
        # Tuned parameters for fast training and high precision/recall on this dataset
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model.fit(X_train, y_train)
    print("Model training completed successfully!")
    return model


def evaluate_model(model, X_test, y_test):
    print("\n[5/5] Evaluating model performance on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Genuine (TN): {cm[0][0]:6d} | False Fraud (FP): {cm[0][1]:6d}")
    print(f"False Genuine (FN): {cm[1][0]:6d} | True Fraud (TP): {cm[1][1]:6d}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud']))

    auprc = average_precision_score(y_test, y_prob)
    print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")


def run_interactive_prediction(model, scaler, X_test, y_test):
    print("\n" + "=" * 60)
    print("      CREDIT CARD FRAUD DETECTION PREDICTION CLI")
    print("=" * 60)
    print("This interface allows you to test the trained model on new inputs.")

    # Store clean unscaled feature lists for input demonstration
    feature_names = list(X_test.columns)

    while True:
        print("\nChoose an option:")
        print("1. Select a random transaction from the test set")
        print("2. Enter a specific transaction index from the test set")
        print("3. Enter custom transaction parameters manually")
        print("4. Exit")

        choice = input("Enter choice (1-4): ").strip()

        if choice == "4":
            print("Exiting prediction interface. Goodbye!")
            break

        elif choice == "1":
            rand_idx = np.random.randint(0, len(X_test))
            sample_x = X_test.iloc[[rand_idx]]
            sample_y = y_test.iloc[rand_idx]

            # Predict
            prob = model.predict_proba(sample_x)[0][1]
            pred = model.predict(sample_x)[0]

            print(f"\n--- Random Transaction Chosen (Test Set Index {rand_idx}) ---")
            print(f"Ground Truth Label: {'FRAUD (1)' if sample_y == 1 else 'Genuine (0)'}")
            print(f"Model Prediction:   {'FRAUD (1)' if pred == 1 else 'Genuine (0)'}")
            print(f"Fraud Probability:  {prob * 100:.4f}%")

        elif choice == "2":
            try:
                idx_str = input(f"Enter transaction index (0 to {len(X_test) - 1}): ").strip()
                idx = int(idx_str)
                if idx < 0 or idx >= len(X_test):
                    print("Index out of range!")
                    continue

                sample_x = X_test.iloc[[idx]]
                sample_y = y_test.iloc[idx]

                # Predict
                prob = model.predict_proba(sample_x)[0][1]
                pred = model.predict(sample_x)[0]

                print(f"\n--- Selected Transaction (Test Set Index {idx}) ---")
                print(f"Ground Truth Label: {'FRAUD (1)' if sample_y == 1 else 'Genuine (0)'}")
                print(f"Model Prediction:   {'FRAUD (1)' if pred == 1 else 'Genuine (0)'}")
                print(f"Fraud Probability:  {prob * 100:.4f}%")
            except ValueError:
                print("Invalid input! Please enter an integer index.")

        elif choice == "3":
            print("\nPlease provide the feature values. The dataset has 30 inputs:")
            print("- Time (raw seconds)")
            # Show list of V1-V28
            print("- V1 to V28 (PCA features, e.g. values typically between -10 and 10)")
            print("- Amount (raw currency amount)")
            print(
                "\nYou can either input them one-by-one or paste a single comma-separated line containing all 30 values in order:")
            print("Format: Time, V1, V2, ..., V28, Amount")

            input_mode = input("Paste comma-separated line? (y/n): ").strip().lower()

            try:
                values = []
                if input_mode == 'y':
                    line = input("Paste the 30 comma-separated values: ").strip()
                    parts = [float(p.strip()) for p in line.split(',')]
                    if len(parts) != 30:
                        print(f"Error: Expected 30 values, but got {len(parts)}.")
                        continue
                    values = parts
                else:
                    # Manually ask for Time, then loop V1-V28, then Amount
                    time_val = float(input("Time (seconds since first transaction): "))
                    values.append(time_val)
                    for i in range(1, 29):
                        val = float(input(f"V{i}: "))
                        values.append(val)
                    amount_val = float(input("Amount (transaction currency value): "))
                    values.append(amount_val)

                # Preprocess:
                # Raw inputs are: [Time, V1, V2, ..., V28, Amount]
                # To match model inputs: [V1, V2, ..., V28, scaled_amount, scaled_time]
                raw_time = values[0]
                pca_features = values[1:29]
                raw_amount = values[29]

                scaled_amount = scaler.transform([[raw_amount]])[0][0]
                scaled_time = scaler.transform([[raw_time]])[0][0]

                # Combine in the correct model feature order: V1..V28, scaled_amount, scaled_time
                model_input = pca_features + [scaled_amount, scaled_time]
                model_input_df = pd.DataFrame([model_input], columns=feature_names)

                # Predict
                prob = model.predict_proba(model_input_df)[0][1]
                pred = model.predict(model_input_df)[0]

                print(f"\n--- Custom Prediction Result ---")
                print(f"Model Prediction:  {'FRAUD (1)' if pred == 1 else 'Genuine (0)'}")
                print(f"Fraud Probability: {prob * 100:.4f}%")
            except Exception as e:
                print(f"Error processing inputs: {e}. Please ensure you enter valid numeric values.")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load and Preprocess Data
    X, y, scaler = load_and_preprocess_data()

    # Stratified split (keeps the same fraud-to-genuine ratio in both sets)
    print("\nSplitting dataset into 80% training and 20% testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Let user select parameters and train the model
    print("\n" + "=" * 40)
    print("       MODEL CONFIGURATION AND TRAINING")
    print("=" * 40)

    print("Choose class imbalance handling technique:")
    print("1. Balanced Class Weights (Adjusts loss function, no data resample) [RECOMMENDED]")
    print("2. Random Undersampling (Downsizes majority class to match minority class)")
    print("3. Random Oversampling (Duplicates minority class to match majority class)")

    imb_choice = input("Enter choice (1-3): ").strip()

    method = "balanced"
    use_class_weights = True

    if imb_choice == "2":
        method = "undersample"
        use_class_weights = False
    elif imb_choice == "3":
        method = "oversample"
        use_class_weights = False

    # Apply class balancing to training set
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train, method=method)

    print("\nChoose classification algorithm:")
    print("1. Logistic Regression (Fast training, linear decision boundary)")
    print("2. Random Forest Classifier (Robust, non-linear ensemble)")

    alg_choice = input("Enter choice (1-2): ").strip()
    algorithm = "logistic_regression"
    if alg_choice == "2":
        algorithm = "random_forest"

    # Train the model
    model = train_model(X_train_balanced, y_train_balanced, algorithm=algorithm, use_class_weights=use_class_weights)

    # Save model and scaler
    print(f"\nSaving model to {MODEL_FILE}...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    print(f"Saving scaler to {SCALER_FILE}...")
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Start Interactive CLI
    run_interactive_prediction(model, scaler, X_test, y_test)


if __name__ == "__main__":
    main()

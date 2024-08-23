from utils.imports import os
from utils.imports import joblib
from utils.imports import accuracy_score

def save_model(model, model_name, folder_name):
    """Saves the model to the specified folder with the given model name."""
    os.makedirs(folder_name, exist_ok=True)
    model_path = os.path.join(folder_name, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model {model_name} saved to {model_path}")

def load_model(model_name, folder_name):
    """Loads the model from the specified folder."""
    model_path = os.path.join(folder_name, f"{model_name}.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model {model_name} loaded from {model_path}")
        return model
    else:
        print(f"Model {model_name} not found in {model_path}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluates the loaded model on test data."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2%}")
    return accuracy
import shap
import numpy as np

def explain_model(model, X_train, X_test, feature_names=None, sample_size=200):
    """
    Generate SHAP explanations for a given model.
    
    model: trained classifier
    X_train: sparse or dense training matrix
    X_test: sparse or dense test matrix
    feature_names: list of feature names for visualization
    sample_size: number of samples to use for explanation (small to avoid memory issues)
    """
    
    # Convert sparse to dense for SHAP if needed
    if hasattr(X_train, "toarray"):
        X_train_sample = X_train[:sample_size].toarray()
    else:
        X_train_sample = X_train[:sample_size]

    if hasattr(X_test, "toarray"):
        X_test_sample = X_test[:sample_size].toarray()
    else:
        X_test_sample = X_test[:sample_size]

    # Create SHAP explainer
    explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_sample)

    # Summary Plot
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names)

    return shap_values

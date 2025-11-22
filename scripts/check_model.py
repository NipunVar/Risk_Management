import joblib, pprint
model_path = r"C:\Users\HP\Documents\Risk_Management\models\credit_risk_prob_model.pkl"

print("\nLoading model:", model_path)
model = joblib.load(model_path)

print("\nModel type:", type(model))
print("Has feature_names_in_:", hasattr(model, "feature_names_in_"))

if hasattr(model, "feature_names_in_"):
    print("\n=== MODEL FEATURE NAMES ===")
    pprint.pprint(list(model.feature_names_in_))
else:
    print("\nModel does not have feature_names_in_. It may not be trained with a pipeline.")

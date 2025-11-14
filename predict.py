import joblib
import pandas as pd

model = joblib.load('demand_model.pkl')
future = pd.DataFrame({'Week': [11, 12, 13, 14, 15]})
preds = model.predict(future)
print("\nNext 5 weeks demand forecast:")
for w, p in zip(future['Week'], preds.round()):
    print(f"Week {w}: {int(p)} units")

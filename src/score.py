import json
import os
import joblib
import pandas as pd

model = None

def init():
    global model
    # Azure automatically sets AZUREML_MODEL_DIR to the path of your registered model
    # Added 'model_output' to the path to match your registered model structure
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_output', 'model.pkl')
    model = joblib.load(model_path)
    
def run(raw_data):
    try:
        # 1. Parse the incoming JSON data
        data = json.loads(raw_data)['data']
        df = pd.DataFrame(data)

        # 2. Feature Selection (Must match Run 3: All Features)
        # This selects only the numeric columns (sentiment, tfidf, metadata)
        X = df.select_dtypes(include=['number'])

        # 3. Predict
        preds = model.predict(X)
        
        # 4. Return results as JSON
        return {"predictions": preds.tolist()}

    except Exception as e:
        return {"error": str(e)}
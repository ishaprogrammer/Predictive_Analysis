from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

app = FastAPI()

# Global variables to store the dataset and model
dataset = None
model = None

# Pydantic model for prediction input
class PredictionInput(BaseModel):
    Spindle_Speed_RPM: float
    Voltage_volts: float
    Torque_Nm: float
    Cutting_kN: float
    Total_Pressure_bar: float
    Average_Temperature: float
    Total_Vibration: float

# Upload Endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global dataset
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.file)
        # Ensure the required columns are present
        required_columns = [
            'Spindle_Speed(RPM)', 'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)',
            'Total_Pressure(bar)', 'Average_Temperature', 'Total_Vibration', 'Downtime'
        ]
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV file must contain the following columns: {required_columns}"
            )
        dataset = df
        return {"message": "File uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Train Endpoint
@app.post("/train")
async def train_model():
    global dataset, model
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")
    
    try:
        # Prepare the data
        X = dataset.drop(columns=['Downtime'])
        y = dataset['Downtime']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Save the model
        joblib.dump(model, 'model.pkl')
        
        return {
            "message": "Model trained successfully",
            "accuracy": accuracy,
            "f1_score": f1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Predict Endpoint
@app.post("/predict")
async def predict_downtime(input_data: PredictionInput):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    try:
        # Prepare input data for prediction
        input_dict = input_data.dict()
        # Map the input keys to match the dataset column names
        input_dict_mapped = {
            'Spindle_Speed(RPM)': input_dict['Spindle_Speed_RPM'],
            'Voltage(volts)': input_dict['Voltage_volts'],
            'Torque(Nm)': input_dict['Torque_Nm'],
            'Cutting(kN)': input_dict['Cutting_kN'],
            'Total_Pressure(bar)': input_dict['Total_Pressure_bar'],
            'Average_Temperature': input_dict['Average_Temperature'],
            'Total_Vibration': input_dict['Total_Vibration']
        }
        input_df = pd.DataFrame([input_dict_mapped])
        
        # Make prediction
        prediction = model.predict(input_df)
        confidence = model.predict_proba(input_df).max()
        
        return {
            "Downtime": "Yes" if prediction[0] == 1 else "No",
            "Confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
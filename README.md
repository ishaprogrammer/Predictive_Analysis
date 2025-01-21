# Machine Downtime Prediction API

## Overview

This project involves building a FastAPI-based RESTful API to predict machine downtime using a preprocessed dataset. The data was first taken from Kaggle, preprocessed, and cleaned, and then used to train a Decision Tree Classifier model. The API provides endpoints for uploading data, training the model, and making predictions.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Endpoints](#endpoints)
6. [Example Workflow](#example-workflow)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features

- **Data Preprocessing**:
  - The dataset was cleaned and preprocessed to ensure it was suitable for training.
- **Exploratory Data Analysis (EDA)**:
  - EDA was performed to understand the data distribution and relationships.
- **FastAPI Implementation**:
  - A RESTful API was built to:
    - Upload a dataset.
    - Train a Decision Tree Classifier.
    - Predict machine downtime based on input features.
- **Model Evaluation**:
  - The model's accuracy and F1-score were evaluated during training.

---

## Installation

To run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Dependencies**:
   Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/srinivasanusuri/optimization-of-machine-downtime).
   - Place the dataset in the `data/` directory.

4. **Run the FastAPI Application**:
   Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

---

## Usage

### Endpoints

The API provides the following endpoints:

1. **`POST /upload`**:
   - Upload a CSV file containing the dataset.
   - The file must contain the following columns:
     - `Spindle_Speed(RPM)`
     - `Voltage(volts)`
     - `Torque(Nm)`
     - `Cutting(kN)`
     - `Total_Pressure(bar)`
     - `Average_Temperature`
     - `Total_Vibration`
     - `Downtime`

2. **`POST /train`**:
   - Train the Decision Tree Classifier on the uploaded dataset.
   - Returns the model's accuracy and F1-score.

3. **`POST /predict`**:
   - Accepts JSON input with the following fields:
     - `Spindle_Speed_RPM`
     - `Voltage_volts`
     - `Torque_Nm`
     - `Cutting_kN`
     - `Total_Pressure_bar`
     - `Average_Temperature`
     - `Total_Vibration`
   - Returns the prediction (`Downtime: Yes/No`) and confidence score.

---

## Example Workflow

1. **Upload the Dataset**:
   ```bash
   curl -X POST -F "file=@data/cleaned_data.csv" http://127.0.0.1:8000/upload
   ```

2. **Train the Model**:
   ```bash
   curl -X POST http://127.0.0.1:8000/train
   ```

3. **Make a Prediction**:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{
     "Spindle_Speed_RPM": 0.93,
     "Voltage_volts": 0.48,
     "Torque_Nm": 0.43,
     "Cutting_kN": 0.84,
     "Total_Pressure_bar": 0.43,
     "Average_Temperature": 0.31,
     "Total_Vibration": 0.56
   }' http://127.0.0.1:8000/predict
   ```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

---

## Acknowledgments

- [Kaggle](https://www.kaggle.com) for the dataset.
- [FastAPI](https://fastapi.tiangolo.com) for the web framework.
- [Scikit-learn](https://scikit-learn.org) for the machine learning tools.

---

For more details, refer to the [API Documentation](http://127.0.0.1:8000/docs).

---

##Author
[Taskin Shaikh]
[Email: ishashaikh154@gmail.com]
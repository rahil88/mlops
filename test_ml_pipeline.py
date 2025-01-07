import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Test constants
DATA_FILE = 'magic04.data'
LR_MODEL_FILE = 'lr_model.pkl'
DT_MODEL_FILE = 'dt_model.pkl'

@pytest.fixture
def load_data():
    cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
    df = pd.read_csv(DATA_FILE)
    df.columns = cols
    df.columns = df.columns.str.strip()
    return df

@pytest.fixture
def prepare_data(load_data):
    df = load_data
    X = df[['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']]
    y = LabelEncoder().fit_transform(df['class'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_data_loading(load_data):
    df = load_data
    assert not df.empty, "Dataframe should not be empty"
    assert 'class' in df.columns, "'class' column should exist in the data"

def test_data_split(prepare_data):
    X_train, X_test, y_train, y_test = prepare_data
    assert len(X_train) > 0 and len(X_test) > 0, "Train and test sets should not be empty"
    assert len(y_train) > 0 and len(y_test) > 0, "Train and test labels should not be empty"

def test_linear_regression_training(prepare_data):
    X_train, X_test, y_train, y_test = prepare_data
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, lr_y_pred)
    assert mse > 0, "Mean Squared Error should be greater than 0"

def test_decision_tree_training(prepare_data):
    X_train, X_test, y_train, y_test = prepare_data
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, dt_y_pred)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"

def test_model_saving_and_loading(prepare_data):
    X_train, _, y_train, _ = prepare_data
    # Train and save models
    lr_model = LinearRegression()
    dt_model = DecisionTreeClassifier(random_state=42)
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    with open(LR_MODEL_FILE, 'wb') as f:
        pickle.dump(lr_model, f)
    with open(DT_MODEL_FILE, 'wb') as f:
        pickle.dump(dt_model, f)
    # Check files exist
    assert os.path.exists(LR_MODEL_FILE), "Linear Regression model file should exist"
    assert os.path.exists(DT_MODEL_FILE), "Decision Tree model file should exist"
    # Load and test models
    with open(LR_MODEL_FILE, 'rb') as f:
        loaded_lr_model = pickle.load(f)
    with open(DT_MODEL_FILE, 'rb') as f:
        loaded_dt_model = pickle.load(f)
    assert loaded_lr_model, "Loaded Linear Regression model should not be None"
    assert loaded_dt_model, "Loaded Decision Tree model should not be None"

def test_clean_up():
    # Clean up model files
    if os.path.exists(LR_MODEL_FILE):
        os.remove(LR_MODEL_FILE)
    if os.path.exists(DT_MODEL_FILE):
        os.remove(DT_MODEL_FILE)
    assert not os.path.exists(LR_MODEL_FILE), "Linear Regression model file should be removed"
    assert not os.path.exists(DT_MODEL_FILE), "Decision Tree model file should be removed"

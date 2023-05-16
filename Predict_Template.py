import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import warnings
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import mne
import joblib
import plotly.graph_objects as go


OUTPUT_NAME = "submit_rf_v1"
OUTPUT_DIR = Path("./output/"+OUTPUT_NAME+"_0.00.csv")
DATA_DIR = Path("./input/")
EDF_DIR = DATA_DIR / "edf_data"
MODEL_DIR = Path("../pickle/"+"Telecommunications_Project-"+OUTPUT_NAME+"_model.pkl")


sample_submission_df = pd.read_csv(DATA_DIR/"sample_submission.csv", parse_dates=[1])
sample_submission_df
train_record_df = pd.read_csv(DATA_DIR/"train_records.csv")
test_record_df = pd.read_csv(DATA_DIR/"test_records.csv")
train_record_df.head()
test_record_df.head()


model = joblib.load(MODEL_DIR)

print("\n==========学習済みモデルを保存しました。==========\n")

val_preds = model.predict(val_df.iloc[:, 3:])
score = accuracy_score(val_df["condition"], val_preds)
print("Accuracy Score："+str(score))
print(classification_report(val_df["condition"], val_preds,))

print("\n==========モデルの評価を完了しました==========\n")

test_df
test_preds = model.predict(test_df.iloc[:, 2:])
sample_submission_df["condition"] = test_preds
sample_submission_df["condition"] = sample_submission_df["condition"].map(ID2LABEL)
sample_submission_df
sample_submission_df["condition"].value_counts()
sample_submission_df.to_csv(OUTPUT_DIR, index=False)

print("\n==========プログラムを完了しました==========\n")
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

len(set(train_record_df["subject_id"].unique()) & set(test_record_df["subject_id"].unique()))
print("訓練被験者数", len(train_record_df["subject_id"].unique()))
print("テスト被験者数", len(test_record_df["subject_id"].unique()))

train_record_df["hypnogram"] = train_record_df["hypnogram"].map(lambda x: str(EDF_DIR/x))
train_record_df["psg"] = train_record_df["psg"].map(lambda x: str(EDF_DIR/x))
test_record_df["psg"] = test_record_df["psg"].map(lambda x: str(EDF_DIR/x))
row = train_record_df.iloc[0]

psg_edf = mne.io.read_raw_edf(row["psg"], preload=False)
type(psg_edf)
psg_edf.info
psg_edf.ch_names
psg_df = psg_edf.to_data_frame()
print(psg_df)
meas_start = psg_edf.info["meas_date"]
meas_start = meas_start.replace(tzinfo=None)
psg_df["meas_time"] = pd.date_range(start=meas_start, periods=len(psg_df), freq=pd.Timedelta(1 / 100, unit="s"))
print(psg_df)

annot = mne.read_annotations(row["hypnogram"])
annot_df = annot.to_data_frame()
print(annot_df)
annot_df["description"].value_counts()
psg_edf = mne.io.read_raw_edf(row["psg"], include=["EEG Fpz-Cz"], verbose=False)
annot = mne.read_annotations(row["hypnogram"])

truncate_start_point = 3600 * 5
truncate_end_point = (len(psg_edf)/100) - (3600 *5)
annot.crop(truncate_start_point, truncate_end_point, verbose=False)
psg_edf.set_annotations(annot, verbose=False, emit_warning=False)

RANDK_LABEL2ID = {
    'Movement time': -1,
    'Sleep stage ?': -1,
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}
events, _ = mne.events_from_annotations(psg_edf, event_id=RANDK_LABEL2ID, chunk_duration=30., verbose=False)
LABEL2ID = {
    'Movement time': -1,
    'Sleep stage ?': -1,
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3/4': 3,
    'Sleep stage R': 4
}
tmax = 30. - 1. / psg_edf.info['sfreq']
epoch = mne.Epochs(raw=psg_edf, events=events, event_id=LABEL2ID, tmin=0, tmax=tmax, baseline=None, verbose=False, on_missing='ignore')

epoch.info["temp"] = {
    "id":row["id"],
    "subject_id":row["subject_id"],
    "night":row["night"],
    "truncate_start_point":truncate_start_point,
    "truncate_end_point":truncate_end_point
}
print(type(epoch))
print(epoch)
epoch_df = epoch.to_data_frame(verbose=False)
print(epoch_df)
print(events)
print(events.shape)
new_meas_date = epoch.info["meas_date"].replace(tzinfo=None) + datetime.timedelta(seconds=epoch.info["temp"]["truncate_start_point"])

epoch_df["meas_time"] = pd.date_range(start=new_meas_date, periods=len(epoch_df), freq=pd.Timedelta(1 / 100, unit="s"))

print(epoch_df)

def epoch_to_df(epoch:mne.epochs.Epochs) -> pd.DataFrame:
    truncate_start_point = epoch.info["temp"]["truncate_start_point"]
    df = epoch.to_data_frame(verbose=False)
    new_meas_date = epoch.info["meas_date"].replace(tzinfo=None) + datetime.timedelta(seconds=truncate_start_point)
    df["meas_time"] = pd.date_range(start=new_meas_date, periods=len(df), freq=pd.Timedelta(1 / 100, unit="s"))
    return df

epoch_to_df(epoch)

label_df = epoch_df.loc[epoch_df.groupby("epoch")["time"].idxmin()][["meas_time"]].reset_index(drop=True)
label_df["condition"] = "Sleep stage W"
label_df["id"] = epoch.info["temp"]["id"]
label_df

def epoch_to_sub_df(epoch_df:pd.DataFrame, id, is_train:bool) -> pd.DataFrame:
    cols = ["id", "meas_time"]
    if is_train:
        cols.append("condition")
    label_df = epoch_df.loc[epoch_df.groupby("epoch")["time"].idxmin()].reset_index(drop=True)
    label_df["id"] = id
    return label_df[cols]

epoch_to_sub_df(epoch_df, epoch.info["temp"]["id"], is_train=True)
test_row = test_record_df.iloc[0]
psg_edf = mne.io.read_raw_edf(test_row["psg"], include=["EEG Fpz-Cz"], verbose=False)

start_psg_date = psg_edf.info["meas_date"]
start_psg_date = start_psg_date.replace(tzinfo=None)

test_start_time = sample_submission_df[sample_submission_df["id"]==test_row["id"]]["meas_time"].min()
test_end_time = sample_submission_df[sample_submission_df["id"]==test_row["id"]]["meas_time"].max()
print(f"psg start: {start_psg_date},  test start: {test_start_time}, test end: {test_end_time}")

truncate_start_point = int((test_start_time - start_psg_date).total_seconds())
truncate_end_point = int((test_end_time - start_psg_date).total_seconds())+30
print(f"event start sencond: {truncate_start_point}, event end second: {truncate_end_point} ")

event_range = list(range(truncate_start_point, truncate_end_point, 30))
events = np.zeros((len(event_range), 3), dtype=int)
events[:, 0] = event_range

events = events * 100

tmax = 30. - 1. / psg_edf.info['sfreq']
epoch = mne.Epochs(raw=psg_edf, events=events, event_id={'Sleep stage W': 0}, tmin=0, tmax=tmax, baseline=None, verbose=False)

epoch.to_data_frame()

def read_and_set_annoation(record_df:pd.DataFrame, include=None, is_test=False) -> List[mne.epochs.Epochs]:
    whole_epoch_data = []
    for row_id, row in tqdm(record_df.iterrows(), total=len(record_df)):
        psg_edf = mne.io.read_raw_edf(row["psg"], include=include, verbose=False)
        if not is_test:
            annot = mne.read_annotations(row["hypnogram"])
            truncate_start_point = 3600 * 5
            truncate_end_point = (len(psg_edf)/100) - (3600 *5)
            annot.crop(truncate_start_point, truncate_end_point, verbose=False)
            psg_edf.set_annotations(annot, emit_warning=False)
            events, _ = mne.events_from_annotations(psg_edf, event_id=RANDK_LABEL2ID, chunk_duration=30., verbose=False)
            event_id = LABEL2ID
        else:
            start_psg_date = psg_edf.info["meas_date"]
            start_psg_date = start_psg_date.replace(tzinfo=None)
            test_start_time = sample_submission_df[sample_submission_df["id"]==row["id"]]["meas_time"].min()
            test_end_time = sample_submission_df[sample_submission_df["id"]==row["id"]]["meas_time"].max()
            truncate_start_point = int((test_start_time - start_psg_date).total_seconds())
            truncate_end_point = int((test_end_time- start_psg_date).total_seconds())+30
            event_range = list(range(truncate_start_point, truncate_end_point, 30))
            events = np.zeros((len(event_range), 3), dtype=int)
            events[:, 0] = event_range
            events = events * 100
            event_id = {'Sleep stage W': 0}
            
        tmax = 30. - 1. / psg_edf.info['sfreq']
        epoch = mne.Epochs(raw=psg_edf, events=events, event_id=event_id, tmin=0, tmax=tmax, baseline=None, verbose=False, on_missing='ignore')
        assert len(epoch.events) * 30 == truncate_end_point - truncate_start_point
        
        epoch.info["temp"] = {
            "id":row["id"],
            "subject_id":row["subject_id"],
            "night":row["night"],
            "age":row["age"],
            "sex":row["sex"],
            "truncate_start_point":truncate_start_point
        }
        whole_epoch_data.append(epoch)
    return whole_epoch_data

train_record_subset_df = train_record_df.sample(n=50).reset_index(drop=True)
train_subset_epoch = read_and_set_annoation(train_record_subset_df, include=["EEG Fpz-Cz"], is_test=False)
test_whole_epoch = read_and_set_annoation(test_record_df, include=["EEG Fpz-Cz"], is_test=True)

sample_events = train_subset_epoch[0].events[:, 2]
sample_epoch_df = train_subset_epoch[0].to_data_frame(verbose=False)

ID2LABEL = {v:k for k, v in LABEL2ID.items()}

go.Figure(
    data=[
        go.Scatter(x=sample_epoch_df["epoch"].unique(), y=list(map(lambda x: ID2LABEL[x], sample_events)))
    ],
    layout=go.Layout(
        yaxis=dict(title="sleep stage"),
        xaxis=dict(title="epoch"),
    )
)

epoch_grouped_df = sample_epoch_df.groupby("epoch").agg({"EEG Fpz-Cz":"mean"}).reset_index()

go.Figure(
    data=[
        go.Scatter(x=epoch_grouped_df["epoch"], y=epoch_grouped_df["EEG Fpz-Cz"]),
    ],
    layout=go.Layout(
        yaxis=dict(title="EEG Fpz-Cz"),
        xaxis=dict(title="epoch"),
    )
)

def eeg_power_band(epochs):
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}
    spectrum = epochs.compute_psd(picks='eeg', fmin=0.5, fmax=30. ,verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    return np.concatenate(X, axis=1)

train_df = []
for epoch in tqdm(train_subset_epoch):
    epoch_df = epoch_to_df(epoch)
    sub_df = epoch_to_sub_df(epoch_df, epoch.info["temp"]["id"], is_train=True)
    feature_df = pd.DataFrame(eeg_power_band(epoch))
    _df = pd.concat([sub_df, feature_df], axis=1)
    _df = _df[~_df["condition"].isin(["Sleep stage ?", "Movement time"])]
    train_df.append(_df)
train_df = pd.concat(train_df).reset_index(drop=True)
train_df["condition"].value_counts()
train_df["condition"] = train_df["condition"].map(LABEL2ID)
test_df = []

for epoch in tqdm(test_whole_epoch):
    epoch_df = epoch_to_df(epoch)
    sub_df = epoch_to_sub_df(epoch_df, epoch.info["temp"]["id"], is_train=False)
    feature_df = pd.DataFrame(eeg_power_band(epoch))
    _df = pd.concat([sub_df, feature_df], axis=1)
    test_df.append(pd.concat([sub_df, feature_df], axis=1))
test_df = pd.concat(test_df)

val_size = int(train_record_df["subject_id"].nunique() * 0.20)
train_all_subjects = train_record_df["subject_id"].unique()
np.random.shuffle(train_all_subjects)
val_subjects = train_all_subjects[:val_size]
val_ids = train_record_df[train_record_df["subject_id"].isin(val_subjects)]["id"]
trn_df = train_df[~train_df["id"].isin(val_ids)]
val_df = train_df[train_df["id"].isin(val_ids)]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(trn_df.iloc[:, 3:], trn_df["condition"])

#joblib.dump(model, MODEL_DIR)
#model = joblib.load(MODEL_DIR)

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
# データの変換

## edfファイルをCSVファイルに変換する
まず、edfファイルをCSVファイルに変換するために、edf4csv.pyを作成していく。

下記のプログラムでは、フォルダに含まれるedfデータをtrainデータとtestデータに分けている。
さらにフォルダ構造を分解していき、edfファイルのメタ情報を取得している
```Python
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

DATA_DIR = Path("./input/")
EDF_DIR = DATA_DIR / "edf_data"

train_record_df = pd.read_csv(DATA_DIR/"train_records.csv")
test_record_df = pd.read_csv(DATA_DIR/"test_records.csv")

# パスを設定
train_record_df["hypnogram"] = train_record_df["hypnogram"].map(lambda x: str(EDF_DIR/x))
train_record_df["psg"] = train_record_df["psg"].map(lambda x: str(EDF_DIR/x))
test_record_df["psg"] = test_record_df["psg"].map(lambda x: str(EDF_DIR/x))

row = train_record_df.iloc[0]

# edfファイルの読み込み
psg_edf = mne.io.read_raw_edf(row["psg"], preload=False)

# 読み込んだデータは、mne.io.edf.edf.RawEDFクラスのインスタンスになります
type(psg_edf)

# infoでメタ情報を表示できます
psg_edf.info
```

上記のプログラムから得られた結果は以下の通り

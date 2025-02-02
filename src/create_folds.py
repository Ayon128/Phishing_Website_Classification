import numpy as np
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.phishing.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    df.loc[:,"id"] = np.arange(1, df.shape[0]+1)
    df.to_csv("../input/train_folds.csv", index=False)
    # print(df.groupby('phishing').size())
    
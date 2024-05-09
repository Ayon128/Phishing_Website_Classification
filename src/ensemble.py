import pandas as pd
import numpy as np
from sklearn import metrics
import model_dispatcher

df = pd.read_csv("../input/train_folds.csv")
lgb_preds = pd.read_csv("../input/lgb_valid_preds.csv").drop(columns = ['phishing','kfold'])
rf_preds = pd.read_csv("../input/rf_valid_preds.csv").drop(columns = ['phishing','kfold'])
cat_preds = pd.read_csv("../input/cat_valid_preds.csv").drop(columns = ['phishing','kfold'])


df = pd.merge(df,lgb_preds ,on=['id'],how='left')
df = pd.merge(df,rf_preds ,on=['id'],how='left')
df = pd.merge(df,cat_preds ,on=['id'],how='left')

pred_cols = [col for col in df.columns if col.find("pred")>=0]
targets = df.phishing.values

pred_dict = {col:df[col].values for col in pred_cols}
pred_rank_dict = {col:df[col].rank().values for col in pred_cols}

print(pred_cols)

## Getting AUC for all models separately 
# for col in pred_cols:
#     auc = metrics.roc_auc_score(targets,df[col].values)
#     print(f"pred_col = {col}; overall_auc ={auc}")

def run_training_stack(pred_df, fold, pred_cols):

    train_df = pred_df[pred_df.kfold!=fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold==fold].reset_index(drop=True)
    
    # xtrain = train_df[pred_cols].values
    # xvalid = valid_df[pred_cols].values
    
    x_train = train_df.drop(["phishing","kfold","id"], axis=1).values
    y_train = train_df.phishing.values
    
    x_valid = valid_df.drop(["phishing","kfold","id"], axis=1).values
    y_valid = valid_df.phishing.values
    
    clf = model_dispatcher.models['lgb']
    
    clf.fit(x_train, y_train)
    
    # preds = clf.predict_proba(xvalid)[:,1]
    
    # auc = metrics.roc_auc_score(valid_df.phishing.values, preds)
    pred = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, pred)
    print(f"fold={fold} acc={accuracy}")

    valid_df.loc[:,"lgb_pred"] = pred
    
    return valid_df, accuracy

if __name__ == "__main__":
    dfs = []
    val_acc = []
    for j in range(10):
        temp_df, acc = run_training_stack(df,j,pred_cols)
        dfs.append(temp_df)
        val_acc.append(acc)
    fin_valid_df = pd.concat(dfs)
    print(np.mean(val_acc))
    
    # fin_valid_df.to_csv("xgb.csv",index=False)
    
    # auc = metrics.roc_auc_score(targets,fin_valid_df.xgb_pred.values)
    
    # print(f"AUC using Xgboost : {auc}")
    print(fin_valid_df.shape)
    
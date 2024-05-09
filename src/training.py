import pandas as pd
import config
from sklearn import metrics
import model_dispatcher


def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)
    test_df = pd.read_csv(config.TESTFILE)
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["phishing","kfold"], axis=1).values
    y_train = df_train.phishing.values
    
    x_valid = df_valid.drop(["phishing","kfold"], axis=1).values
    y_valid = df_valid.phishing.values
    
    X_test = test_df.drop("phishing", axis=1).values
    y_test = test_df.phishing.values
    
    
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    # pred = clf.predict_proba(x_valid)[:,1]
    
    # auc = metrics.roc_auc_score(y_valid, pred)
    pred = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, pred)
    
    print(f"fold = {fold} Acc = {accuracy}")
    
    df_valid.loc[:,f'{model}_pred'] = pred
    
    return df_valid[['id','phishing','kfold',f'{model}_pred']]
    

def run_all(model):
    dfs = []
    for j in range(10):
        temp_df = run(j, model)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    
    print(fin_valid_df.shape)
    
    fin_valid_df.to_csv(f"../input/{model}_valid_preds.csv", index=False)

if __name__ == "__main__":
    run_all("xgb")
    run_all("rf")
    run_all("cat")
    
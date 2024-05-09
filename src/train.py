import joblib
import pandas as pd
import config
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    test_df = pd.read_csv(config.TESTFILE)
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["phishing","kfold","id"], axis=1).values
    y_train = df_train.phishing.values
    
    x_valid = df_valid.drop(["phishing","kfold","id"], axis=1).values
    y_valid = df_valid.phishing.values
    
    X_test = test_df.drop("phishing", axis=1).values
    y_test = test_df.phishing.values
    
    
    # model = CatBoostClassifier(
    #     iterations=500, 
    #     random_seed=42,
    #     learning_rate=0.5,
    #     custom_loss=['Accuracy']
    # # )
    model = XGBClassifier(
        n_estimators=500,     # Number of gradient boosted trees. Equivalent to number of boosting rounds.
        learning_rate=0.1,   # Boosting learning rate (xgb’s “eta”)
        random_state=42,     # Random number seed.
        max_depth=20,         # Maximum tree depth for base learners.
        min_child_weight=1,  # Minimum sum of instance weight(hessian) needed in a child.
        gamma=0,             # Minimum loss reduction required to make a further partition on a leaf node of the tree.
        subsample=1,         # Subsample ratio of the training instance.
        colsample_bytree=1,  # Subsample ratio of columns when constructing each tree.
        objective='binary:logistic',  # Specify the learning task and the corresponding learning objective.
        nthread=4,           # Number of parallel threads used to run xgboost.
        scale_pos_weight=1,  # Balancing of positive and negative weights.
        seed=42,             # Random number seed. 
        reg_alpha=0,         # L1 regularization term on weights
        reg_lambda=1,        # L2 regularization term on weights
        use_label_encoder=False,  # To avoid a warning message
        eval_metric='logloss'  # Evaluation metrics for validation data
    )
    
    # model = RandomForestClassifier(
    #     n_estimators=200,
    #     random_state=42,
    #     n_jobs=-1
    # )
    model.fit(
        x_train,
        y_train,
        # eval_set=(x_valid, y_valid),
        # verbose=False  # Set verbose=False to suppress progress output
    )


# Initialize the base estimator - a decision tree with max_depth=2
    # model = lgb.LGBMClassifier(
    #     n_estimators=500,
    #     num_leaves=31,
    #     learning_rate=0.1,
    #     random_state=42
    # )
    # model.fit(x_train, y_train)
    # print('Tree count: ' + str(model.tree_count_))
    preds = model.predict(x_valid)
    preds_test = model.predict(X_test)
    accuracy1 = metrics.accuracy_score(y_valid, preds)
    accuracy = metrics.accuracy_score(y_test, preds_test)
    print(f"Fold={fold}, Val_accuracy={accuracy1}, Test_accuracy={accuracy}")
    return accuracy1, accuracy


if __name__ == "__main__":
    val_acc = []
    test_acc = []
    for i in range(10):
        acc, acc1 = run(fold = i)
        val_acc.append(acc)
        test_acc.append(acc1)
   
    print(f"Mean Accuracy={sum(val_acc)/len(val_acc)}")
    print(f"Mean Test Accuracy={sum(test_acc)/len(test_acc)}")
    
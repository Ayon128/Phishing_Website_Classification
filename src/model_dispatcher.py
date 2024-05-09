from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

models = {
    "rf": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    
    "cat": CatBoostClassifier(
        iterations=500, 
        random_seed=42,
        learning_rate=0.5,
        custom_loss=['Accuracy']
    ),
    
    "xgb": XGBClassifier(
        n_estimators=500,    
        learning_rate=0.1,  
        random_state=42,    
        max_depth=20,         
        min_child_weight=1,  
        gamma=0,             
        subsample=1,         
        colsample_bytree=1,  
        objective='binary:logistic',  
        nthread=4,           
        scale_pos_weight=1,  
        seed=42,             
        reg_alpha=0,        
        reg_lambda=1,       
        use_label_encoder=False, 
        eval_metric='logloss' 
        ),
    
    "lgb": lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.1,
        random_state=42
    )
}
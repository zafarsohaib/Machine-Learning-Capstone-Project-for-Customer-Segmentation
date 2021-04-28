import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator

import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.pipeline import Pipeline

SEED = 42


def selected_features(df):
    ''' reducing to selected number of features
    '''
    
    columns_to_keep = ['ALTERSKATEGORIE_GROB', 'ANZ_KINDER', 'ANZ_STATISTISCHE_HAUSHALTE', 'CJT_KATALOGNUTZER', 
    'D19_KONSUMTYP','D19_BANKEN_ONLINE_DATUM', 'D19_BANKEN_REST', 'D19_BILDUNG', 
    'D19_GESAMT_ANZ_24', 'D19_KONSUMTYP_MAX', 'D19_REISEN', 'D19_SOZIALES', 'D19_TIERARTIKEL', 
    'DECADE', 'FINANZ_SPARER', 'FINANZ_UNAUFFAELLIGER', 'FINANZ_VORSORGER', 
    'GEBAEUDETYP', 'GFK_URLAUBERTYP', 'HH_EINKOMMEN_SCORE', 
    'KBA05_ALTER4', 'KBA05_ANHANG', 'KBA05_ANTG1', 'KBA05_AUTOQUOT', 'KBA05_CCM1', 
    'KBA05_GBZ', 'KBA05_KRSKLEIN', 'KBA05_KRSZUL', 'KBA05_KW1', 'KBA05_KW2', 'KBA05_MAXAH', 
    'KBA05_MAXSEG', 'KBA05_MOD1', 'KBA05_MOD4', 'KBA05_SEG01', 'KBA05_SEG2', 'KBA05_VORB0', 'KBA05_ZUL4', 
    'KBA13_ANZAHL_PKW', 'KBA13_AUDI', 'KBA13_BJ_1999', 'KBA13_BJ_2000', 'KBA13_CCM_2500', 'KBA13_CCM_3000', 
    'KBA13_KRSSEG_OBER', 'KBA13_KRSSEG_VAN', 'KBA13_KRSZUL_NEU', 'KBA13_KW_0_60', 'KBA13_KW_61_120', 'KBA13_MOTOR', 
    'KBA13_SEG_KOMPAKTKLASSE', 'KBA13_SEG_SPORTWAGEN', 
    'KBA13_SITZE_5', 'KBA13_SITZE_6', 'KBA13_VORB_0', 'MOBI_REGIO', 'OST_WEST_KZ', 'PLZ8_ANTG1', 'PLZ8_ANTG3', 'REGIOTYP', 
    'RT_KEIN_ANREIZ', 
    'SEMIO_ERL', 'SEMIO_KRIT', 'SEMIO_LUST', 'SEMIO_MAT', 'SEMIO_PLICHT', 'SEMIO_REL', 'SEMIO_SOZ', 
    'UNGLEICHENN_FLAG', 'VERS_TYP', 'VHA', 'VHN', 'VK_ZG11', 'WEALTH', 'WOHNDAUER_2008', 'ZABEOTYP', 'ZABEOTYP_3'] 
    
    selected_features = ['D19_SOZIALES',
                         'GEBURTSJAHR',
                         'GFK_URLAUBERTYP',
                         'CJT_GESAMTTYP',
                         'VERDICHTUNGSRAUM',
                         'D19_KONSUMTYP_MAX',
                         'HH_EINKOMMEN_SCORE',
                         'KBA05_ZUL4',
                         'ALTERSKATEGORIE_FEIN',
                         'KBA13_KW_120',
                         'KBA13_VORB_0',
                         'D19_VERSI_ONLINE_QUOTE_12',
                         'D19_TELKO_ONLINE_QUOTE_12']
    selected_df = df[df.columns[df.columns.isin(selected_features)]].astype('float')
    selected_df.index = df.index
    
    return selected_df

def preprocess(df, dias, limit=30):
    """ Pre-process data till feature engineering
    """
    convert_special_char_to_nan(df)
    covert_unknown_to_nan(df, dias)
    percent_nan_df = percent_missing_df(df)
    columns_over_limit = list(percent_nan_df[percent_nan_df['percent'] > limit].index)
    print("Dropping columns with high percentage of missing values", columns_over_limit)
    df.drop(columns_over_limit, axis=1, inplace=True)
    df_eng = feature_eng(df)
    return df_eng

def convert_special_char_to_nan(df):
    """ Convert special character to np.nan
    """
    special_chars = ['X', 'XX', ' ', '']
    columns_list = ['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015' ]
    for col in columns_list:
        df[col].replace(special_chars, np.nan, inplace=True)
        
def covert_unknown_to_nan(df, dias):
    """ Convert unknown values like 0, -1, 9 to np.nan 
    """
    unknowns = dias['Meaning'].where(dias.Meaning.str.contains("unknown")).value_counts().index
    dias = dias[dias['Meaning'].isin(unknowns)]
    for row in dias.iterrows():
        missing_val = row[1]['Value']
        attr = row[1]['Attribute']
        if attr not in df: 
            continue
        if isinstance(missing_val, int):
            df[attr].replace(missing_val, np.nan, inplace=True)
        elif isinstance(missing_val, str):
            df[attr].replace(eval("["+missing_val+"]"), np.nan, inplace=True)
            
def percent_missing_df(df):
    percent = df.isnull().sum()*100/len(df.index)
    new_df = pd.DataFrame({'percent': percent}, index=df.columns)
    return new_df

def feature_eng_old(df):
    """ Feature Engineering
    """
    col_drop = []
    # label encoding CAMEO_DEU_2015 (one-hot encoding consumes a lot of memory because of too many unique labels)
    if 'CAMEO_DEU_2015' in df:
        print("label encoding CAMEO_DEU_2015")
        cameo_fill = df['CAMEO_DEU_2015'].value_counts().idxmax()
        df['CAMEO_DEU_2015'] = df['CAMEO_DEU_2015'].fillna(cameo_fill)
        data = np.array(df['CAMEO_DEU_2015'])
        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(data)
        df['CAMEO_DEU_2015'] = encoded_values
    
    if 'D19_LETZTER_KAUF_BRANCHE' in df:
        print("label encoding D19_LETZTER_KAUF_BRANCHE")
        letzer_fill = df['D19_LETZTER_KAUF_BRANCHE'].value_counts().idxmax()
        df['D19_LETZTER_KAUF_BRANCHE'] = df['D19_LETZTER_KAUF_BRANCHE'].fillna(letzer_fill)
        data = np.array(df['D19_LETZTER_KAUF_BRANCHE'])
        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(data)
        df['D19_LETZTER_KAUF_BRANCHE'] = encoded_values
    

    # Convert CAMEO_DEUG_2015 str to float
    if 'CAMEO_DEUG_2015' in df:
        print("Convert CAMEO_DEUG_2015 str to float")
        df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].apply(lambda x: float(x))
    
    if 'OST_WEST_KZ' in df:
        
        # Mapping OST_WEST_KZ categorical
        # W = 1
        # O = 0
        print("Mapping OST_WEST_KZ categorical")
        ost_west_map = {'W': 0, 'O': 1}
        df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map(ost_west_map)
                                                                            
    # Splitting CAMEO_INTL_2015 Feature into 2 features FAMILY and WEALTH
    if 'CAMEO_INTL_2015' in df:
        print("Splitting CAMEO_INTL_2015 Feature into 2 features FAMILY and WEALTH")
        # WEALTH
        # 1: Wealthy Households
        # 2: Prosperous Households
        # 3: Comfortable Households
        # 4: Less Affluent Households
        # 5: Poorer Households
        df['WEALTH'] = df['CAMEO_INTL_2015'].apply(lambda x: np.floor_divide(int(x), 10) if isinstance(x, str) else np.nan)
        # FAMILY
        # 1: Pre-Family Couples & Singles
        # 2: Young Couples With Children
        # 3: Families With School Age Children
        # 4: Older Families & Mature Couples
        # 5: Elders In Retirement
        df['FAMILY'] = df['CAMEO_INTL_2015'].apply(lambda x: np.mod(int(x), 10) if isinstance(x, str) else np.nan)
        col_drop.append('CAMEO_INTL_2015')
  
    # Splitting LP_LEBENSPHASE_FEIN into two features INCOME and AGE
    if 'LP_LEBENSPHASE_FEIN' in df:
        print("Splitting LP_LEBENSPHASE_FEIN into two features INCOME and AGE")
        # INCOME
        # 1: low
        # 2: average
        # 3: wealthy
        # 4: top
        income_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 3,
                      11: 2, 12: 2, 13: 4, 14: 2, 15: 1, 16: 2, 17: 2, 18: 3, 19: 3, 20: 4,
                      21: 1, 22: 2, 23: 3, 24: 1, 25: 2, 26: 2, 27: 2, 28: 4, 29: 1, 30: 2,
                      31: 1, 32: 2, 33: 2, 34: 2, 35: 4, 36: 2, 37: 2, 38: 2, 39: 4, 40: 4}
        df['INCOME'] = df['LP_LEBENSPHASE_FEIN'].map(income_map)
        # AGE
        # 1: young
        # 2: middle
        # 3: advanced
        # 4: retirement
        age_map = {1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 4, 7: 3, 8: 4, 9: 2, 10: 2,
                  11: 3, 12: 4, 13: 3, 14: 1, 15: 3, 16: 3, 17: 2, 18: 1, 19: 3, 20: 3,
                  21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 1, 30: 1,
                  31: 3, 32: 3, 33: 1, 34: 1, 35: 1, 36: 3, 37: 3, 38: 4, 39: 2, 40: 4}
        df['AGE'] = df['LP_LEBENSPHASE_FEIN'].map(age_map)
        col_drop.append('LP_LEBENSPHASE_FEIN')
    
   
    # Splitting PRAEGENDE_JUGENDJAHRE into two DECADE and MOVEMENT
    if 'PRAEGENDE_JUGENDJAHRE' in df:
        print("Splitting PRAEGENDE_JUGENDJAHRE into two DECADE and MOVEMENT")
        # DECADE
        decade_map = {1: 40, 2:40, 3: 50, 4:50, 5:60, 6:60, 7:60, 8:70, 9:70, 10:80, 
                      11:80, 12:80, 13:80, 14:90, 15:90}
        df['DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].map(decade_map)
        # MOVEMENT
        # 0: Mainstream
        # 1: Avantgarde
        movement_map = {1: 0, 2:1, 3: 0, 4:1, 5:0, 6:1, 7:1, 8:0, 9:1, 10:0, 
                      11:1, 12:0, 13:1, 14:0, 15:1}
        df['MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement_map)
        col_drop.append('PRAEGENDE_JUGENDJAHRE')

    if 'EINGEFUEGT_AM' in df:
        # Get just the registeration year
        print("Get the registeration year from EINGEFUEGT_AM")
        df['EINGEFUEGT_YEAR'] = pd.to_datetime(df['EINGEFUEGT_AM']).dt.year
        col_drop.append('EINGEFUEGT_AM')
    
    print("Dropping unwanted columns", col_drop)
    df.drop(columns=col_drop, axis=1, inplace=True)
    
    print("Imputing missing values with most frequent value")
    #Impute the missing values with the most frequent value of the column
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed_df = pd.DataFrame(imputer.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index
    
    return imputed_df

def feature_eng(df):
    """ Feature Engineering
    """
    col_drop = []
    # label encoding CAMEO_DEU_2015 (one-hot encoding consumes a lot of memory because of too many unique labels)
    if 'CAMEO_DEU_2015' in df:
        print("label encoding CAMEO_DEU_2015")
        cameo_fill = df['CAMEO_DEU_2015'].value_counts().idxmax()
        df['CAMEO_DEU_2015'] = df['CAMEO_DEU_2015'].fillna(cameo_fill)
        data = np.array(df['CAMEO_DEU_2015'])
        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(data)
        df['CAMEO_DEU_2015'] = encoded_values
    """
    if 'D19_LETZTER_KAUF_BRANCHE' in df:
        print("label encoding D19_LETZTER_KAUF_BRANCHE")
        letzer_fill = df['D19_LETZTER_KAUF_BRANCHE'].value_counts().idxmax()
        df['D19_LETZTER_KAUF_BRANCHE'] = df['D19_LETZTER_KAUF_BRANCHE'].fillna(letzer_fill)
        data = np.array(df['D19_LETZTER_KAUF_BRANCHE'])
        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(data)
        df['D19_LETZTER_KAUF_BRANCHE'] = encoded_values
    """
    col_drop.append('D19_LETZTER_KAUF_BRANCHE')
    
    # Convert CAMEO_DEUG_2015 str to float
    if 'CAMEO_DEUG_2015' in df:
        print("Convert CAMEO_DEUG_2015 str to float")
        df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].apply(lambda x: float(x))
    
    if 'OST_WEST_KZ' in df:
        
        # Mapping OST_WEST_KZ categorical
        # W = 1
        # O = 0
        print("Mapping OST_WEST_KZ categorical")
        ost_west_map = {'W': 0, 'O': 1}
        df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map(ost_west_map)
                                                                            
    # Splitting CAMEO_INTL_2015 Feature into 2 features FAMILY and WEALTH
    if 'CAMEO_INTL_2015' in df:
        print("Splitting CAMEO_INTL_2015 Feature into 2 features FAMILY and WEALTH")
        # WEALTH
        # 1: Wealthy Households
        # 2: Prosperous Households
        # 3: Comfortable Households
        # 4: Less Affluent Households
        # 5: Poorer Households
        df['WEALTH'] = df['CAMEO_INTL_2015'].apply(lambda x: np.floor_divide(int(x), 10) if isinstance(x, str) else np.nan)
        # FAMILY
        # 1: Pre-Family Couples & Singles
        # 2: Young Couples With Children
        # 3: Families With School Age Children
        # 4: Older Families & Mature Couples
        # 5: Elders In Retirement
        df['FAMILY'] = df['CAMEO_INTL_2015'].apply(lambda x: np.mod(int(x), 10) if isinstance(x, str) else np.nan)
        col_drop.append('CAMEO_INTL_2015')
  
    # Splitting LP_LEBENSPHASE_FEIN into two features INCOME and AGE
    if 'LP_LEBENSPHASE_FEIN' in df:
        print("Splitting LP_LEBENSPHASE_FEIN into two features INCOME and AGE")
        # INCOME
        # 1: low
        # 2: average
        # 3: wealthy
        # 4: top
        income_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 3,
                      11: 2, 12: 2, 13: 4, 14: 2, 15: 1, 16: 2, 17: 2, 18: 3, 19: 3, 20: 4,
                      21: 1, 22: 2, 23: 3, 24: 1, 25: 2, 26: 2, 27: 2, 28: 4, 29: 1, 30: 2,
                      31: 1, 32: 2, 33: 2, 34: 2, 35: 4, 36: 2, 37: 2, 38: 2, 39: 4, 40: 4}
        df['INCOME'] = df['LP_LEBENSPHASE_FEIN'].map(income_map)
        # AGE
        # 1: young
        # 2: middle
        # 3: advanced
        # 4: retirement
        age_map = {1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 4, 7: 3, 8: 4, 9: 2, 10: 2,
                  11: 3, 12: 4, 13: 3, 14: 1, 15: 3, 16: 3, 17: 2, 18: 1, 19: 3, 20: 3,
                  21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 1, 30: 1,
                  31: 3, 32: 3, 33: 1, 34: 1, 35: 1, 36: 3, 37: 3, 38: 4, 39: 2, 40: 4}
        df['AGE'] = df['LP_LEBENSPHASE_FEIN'].map(age_map)
        col_drop.append('LP_LEBENSPHASE_FEIN')
    
   
    # Splitting PRAEGENDE_JUGENDJAHRE into two DECADE and MOVEMENT
    if 'PRAEGENDE_JUGENDJAHRE' in df:
        print("Splitting PRAEGENDE_JUGENDJAHRE into two DECADE and MOVEMENT")
        # DECADE
        decade_map = {1: 40, 2:40, 3: 50, 4:50, 5:60, 6:60, 7:60, 8:70, 9:70, 10:80, 
                      11:80, 12:80, 13:80, 14:90, 15:90}
        df['DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].map(decade_map)
        # MOVEMENT
        # 0: Mainstream
        # 1: Avantgarde
        movement_map = {1: 0, 2:1, 3: 0, 4:1, 5:0, 6:1, 7:1, 8:0, 9:1, 10:0, 
                      11:1, 12:0, 13:1, 14:0, 15:1}
        df['MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement_map)
        col_drop.append('PRAEGENDE_JUGENDJAHRE')

    if 'EINGEFUEGT_AM' in df:
        # Get just the registeration year
        print("Get the registeration year from EINGEFUEGT_AM")
        df['EINGEFUEGT_YEAR'] = pd.to_datetime(df['EINGEFUEGT_AM']).dt.year
        col_drop.append('EINGEFUEGT_AM')
    
    print("Dropping unwanted columns", col_drop)
    df.drop(columns=col_drop, axis=1, inplace=True)
    
    print("Imputing missing values with most frequent value")
    #Impute the missing values with the most frequent value of the column
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed_df = pd.DataFrame(imputer.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index
    
    return imputed_df

def feature_scaling(df, scaling='minmax'):
    """ Perform Min-Max Scaling
    """
    if scaling == 'minmax':
        scaler = MinMaxScaler()
    elif scaling == 'standard':
        scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df))
    scaled_df.columns = df.columns
    scaled_df.index = df.index
    return scaled_df

def plot_feature_importance(model, features, n=10):
    """ Plot n number of features from the model
    """
    sorted_idx = model.feature_importances_.argsort()
    df = pd.DataFrame()
    df['features'] = features
    df['importance'] = model.feature_importances_
    df_sorted = df.sort_values('importance', ascending=False)
    df_sorted[:n].plot.barh(x='features', y='importance')
    print(df_sorted[:n])
    plt.xlabel("Feature Importance")
    return df_sorted[:n]

    
def create_baseline_models():
    """ Create Baseline Models using differing modeling algorithms
    """
    models = []
    #models.append(('SVC', SVC(random_state=SEED)))
    models.append(('LGBM', lgb.LGBMClassifier(random_state=SEED)))
    models.append(('GB', GradientBoostingClassifier(random_state=SEED)))
    models.append(('RF', RandomForestClassifier(n_estimators=250, random_state=SEED)))
    models.append(('LogR', LogisticRegression(solver='liblinear', random_state=SEED)))
    models.append(('MLP', MLPClassifier(random_state=SEED)))
    models.append(('XGB', xgb.XGBClassifier(random_state=SEED, use_label_encoder =False, eval_metric='auc')))   
    return models

def compare_models(data, response, models, plot=False):
    """ Compare Models by using cross-validation
    """
    results = []
    names = []
    for name, model in models:
        start = time.time()
        cv_results = cross_val_score(model, data, response, cv=5, scoring='roc_auc', n_jobs=1)
        stop = time.time()
        results.append(np.round(cv_results.mean(), 4))
        names.append(name)
        log = "Model: {}; Mean-AUC: {}; Std: {}; Training time: {} min".format(name, 
                                                        np.round(cv_results.mean(), 4), 
                                                        np.round(cv_results.std(), 4),
                                                        np.round((stop - start)/60, 2)
                                                       )
        print(log)
        if plot:
            train_sizes, train_scores, test_scores = learning_curve(
                model, data, response, cv=5, scoring = 'roc_auc', train_sizes=np.linspace(.1, 1.0, 10), n_jobs=1)

            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            print("ROC-AUC train score = {}".format(train_scores_mean[-1].round(2)))
            print("ROC-AUC cross-validation score = {}\n".format(test_scores_mean[-1].round(2)))
            plt.grid()
            plt.title("Learning Curve")
            plt.xlabel("% of Training Data")
            plt.ylabel("Score")
            plt.plot(np.linspace(.1, 1.0, 10)*100, train_scores_mean, 'o-', color="g", label="Training score")
            plt.plot(np.linspace(.1, 1.0, 10)*100, test_scores_mean, 'o-', color="r", label="Cross-validation score")
            plt.yticks(np.arange(0.45, 1.02, 0.05))
            plt.xticks(np.arange(0., 100.05, 10))
            plt.show()
        
        
    return names, results


def get_scaled_models(scaling):
    """ Model Pipelining with scaling option of 'Standard' or 'MinMax'
    """
    
    if scaling == 'minmax':
        scaler = MinMaxScaler()
    elif scaling == 'standard':
        scaler = StandardScaler()
        
    pipelines = []
    pipelines.append((scaling+'LGBM', Pipeline([('Scaler', scaler), ('LGBM', lgb.LGBMClassifier(random_state=SEED))])))
    pipelines.append((scaling+'GB', Pipeline([('Scaler', scaler), ('GB', GradientBoostingClassifier(random_state=SEED))])))
    pipelines.append((scaling+'RF', Pipeline([('Scaler', scaler), ('RF', RandomForestClassifier(n_estimators=250, random_state=SEED))])))
    #pipelines.append((scaling+'SVC', Pipeline([('Scaler', scaler), ('SVC', SVC(random_state=SEED))])))
    pipelines.append((scaling+'LogR', Pipeline([('Scaler', scaler), ('LogR', LogisticRegression(solver='liblinear', random_state=SEED))])))
    pipelines.append((scaling+'MLP', Pipeline([('Scaler', scaler), ('MLP', MLPClassifier(random_state=SEED))])))   
    pipelines.append((scaling+'XGB', Pipeline([('Scaler', scaler), ('XGB', xgb.XGBClassifier(use_label_encoder =False, eval_metric='auc', random_state=SEED))])))

    return pipelines

def feature_selection(feature_list, data, response):
    """ Returns the list of features which increase the LGBM model performance
    """
    model = lgb.LGBMClassifier(random_state=SEED)
    selected_features = []
    highest_score = 0
    for feature in feature_list:
        columns_to_keep = selected_features + [feature]
        scaled_df = feature_scaling(data, 'minmax')
        scaled_df = scaled_df[scaled_df.columns[scaled_df.columns.isin(columns_to_keep)]].astype('float')
        cv_results = cross_val_score(model, scaled_df, response, cv=5, scoring='roc_auc', n_jobs=1)
        if cv_results.mean() > highest_score:
            highest_score = cv_results.mean()
            selected_features.append(feature)
        print('Feature: {}, Score: {}, Best Score: {}'.format(feature, cv_results.mean(), highest_score))
    return selected_features, highest_score
    
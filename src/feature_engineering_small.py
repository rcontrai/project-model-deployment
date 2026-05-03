# Classes et fonctions permettant de préparer les données

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer # NB: version de scikit-learn : 1.8.0
from sklearn.impute import SimpleImputer#, IterativeImputer
# from sklearn.linear_model import BayesianRidge
# from sklearn.linear_model import LinearRegression

def load_app_file(path:str)->tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge le fichier app_train.csv ou app_test.csv en effectuant une première préparation
    """
    root, filext = os.path.splitext(path)
    if filext == ".csv":
        app = pd.read_csv(path, index_col="SK_ID_CURR")
    elif filext == ".parquet":
        app = pd.read_parquet(path)
        app.set_index("SK_ID_CURR", inplace=True)
    target = app["TARGET"]
    app.drop("TARGET", axis=1, inplace=True)
    return app, target

INPUTS_APP = ["SK_ID_CURR", 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE']
INPUTS_BUREAU = ["SK_ID_CURR", "SK_ID_BUREAU", 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'DAYS_CREDIT', 'CREDIT_ACTIVE']
INPUTS_PREV_APP = ["SK_ID_CURR", "SK_ID_PREV",'AMT_APPLICATION', 'DAYS_DECISION']

def shrink_app(app:pd.DataFrame, target:pd.Series=None)->pd.DataFrame:
    """
    Optimise la taille de la table application en la restreignant aux colonnes utilisées pour l'inférence 
    et en choisissant des types et des encodages plus économes en mémoire.

    Nécessaire avant d'appliquer le pipeline de préparation.
    """
    app_smaller = app[INPUTS_APP].copy()
    if not (target is None):
        app_smaller["TARGET"] = target.astype('bool')
    # Encodage des catégories pour gagner de la place (et du temps de transmission)
    name_contract_type_dict = {"Cash loans":False, "Revolving loans":True}
    code_gender_dict = {"F":0, "M":1, "XNA":2} #ce n'est pas le rôle de ce composant de gérer les valeurs aberrantes
    flag_own_car_dict = {"N":False, "Y":True}
    name_eduction_type_dict = {"Lower secondary" : 0,               # Encodage ordinal
                            "Secondary / secondary special" : 1,
                            "Incomplete higher" : 2,
                            "Higher education" : 3,
                            "Academic degree" : 4}
    name_family_status_dict = {'Married': 0,             # Encodage arbitraire, à réviser avant de passer au modèle
                            'Single / not married': 1,
                            'Civil marriage': 2,
                            'Separated': 3,
                            'Widow': 4,
                            'Unknown': 5}
    app_smaller["NAME_CONTRACT_TYPE"] = app_smaller["NAME_CONTRACT_TYPE"].map(name_contract_type_dict)
    app_smaller["CODE_GENDER"] = app_smaller["CODE_GENDER"].map(code_gender_dict)
    app_smaller["FLAG_OWN_CAR"] = app_smaller["FLAG_OWN_CAR"].map(flag_own_car_dict)
    app_smaller["NAME_EDUCATION_TYPE"] = app_smaller["NAME_EDUCATION_TYPE"].map(name_eduction_type_dict)
    app_smaller["NAME_FAMILY_STATUS"] = app_smaller["NAME_FAMILY_STATUS"].map(name_family_status_dict)
    # Inversion des temps
    app_smaller["DAYS_BIRTH"] = - app_smaller["DAYS_BIRTH"]
    app_smaller["DAYS_EMPLOYED"] = - app_smaller["DAYS_EMPLOYED"]
    app_smaller["DAYS_ID_PUBLISH"] = - app_smaller["DAYS_ID_PUBLISH"]
    app_smaller["DAYS_LAST_PHONE_CHANGE"] = - app_smaller["DAYS_LAST_PHONE_CHANGE"]
    # Imputation par la médiane pour une NaN récalcitrant qui empêche la conversion d'une colonne en entier
    days_last_phone_change_mis_val = app_smaller["DAYS_LAST_PHONE_CHANGE"].isna() #littéralement 1 seul exemple
    app_smaller.loc[days_last_phone_change_mis_val, "DAYS_LAST_PHONE_CHANGE"] = 757
    # Optimisation des types
    small_ints = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"]
    app_smaller[small_ints] = app_smaller[small_ints].astype("uint8")
    ints = ["SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]
    app_smaller[ints] = app_smaller[ints].astype("int32")
    floats = app_smaller.select_dtypes(float).columns
    app_smaller[floats] = app_smaller[floats].astype("float32")
    return app_smaller

def shrink_bureau(bureau:pd.DataFrame)->pd.DataFrame:
    """
    Optimise la taille de la table bureau en la restreignant aux colonnes utilisées pour l'inférence 
    et en choisissant des types et des encodages plus économes en mémoire.

    Nécessaire avant d'appliquer le pipeline de préparation.
    """
    bureau_smaller = bureau[INPUTS_BUREAU].copy()
    # On a besoin que la colonne SK_ID_BUREAU existe mais son contenu importe peu (du moment qu'elle n'est pas vide)
    bureau_smaller["SK_ID_BUREAU"] = False
    # Encodage des catégories
    credit_active_dict = {"Closed" : 0, #Encodage arbitraire, à reprendre avant de le passer au modèle
                        "Active" : 1,
                        "Sold" : 2,
                        "Bad debt" : 3}
    bureau_smaller["CREDIT_ACTIVE"] = bureau_smaller["CREDIT_ACTIVE"].map(credit_active_dict)
    # Inversion des temps
    bureau_smaller["DAYS_CREDIT"] = - bureau_smaller["DAYS_CREDIT"]
    # optimisation des types
    bureau_smaller["CREDIT_ACTIVE"] = bureau_smaller["CREDIT_ACTIVE"].astype("uint8")
    ints = ["SK_ID_CURR", "DAYS_CREDIT"]
    bureau_smaller[ints] = bureau_smaller[ints].astype("int32")
    bureau_smaller[["AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM"]] = bureau_smaller[["AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM"]].astype("float32")
    return bureau_smaller

def shrink_prev_app(prev_app:pd.DataFrame)->pd.DataFrame:
    """
    Optimise la taille de la table previous_application en la restreignant aux colonnes utilisées pour l'inférence 
    et en choisissant des types et des encodages plus économes en mémoire.

    Nécessaire avant d'appliquer le pipeline de préparation.
    """
    prev_app_smaller = prev_app[INPUTS_PREV_APP].copy()
    # On a besoin que la colonne SK_ID_PREV existe mais son contenu importe peu (du moment qu'elle n'est pas vide)
    prev_app_smaller["SK_ID_PREV"] = False
    # Drop des demandes incomplètes (0.3%), qui sont aussi présentes en version complète dans la table
    prev_app_smaller.drop(prev_app.index[prev_app["NFLAG_LAST_APPL_IN_DAY"] == 0], axis=0, inplace=True)
    # Inversion des temps
    prev_app_smaller["DAYS_DECISION"] = - prev_app_smaller["DAYS_DECISION"]
    # Optimisation des types
    ints = prev_app_smaller.select_dtypes(int).columns
    prev_app_smaller[ints] = prev_app_smaller[ints].astype("int32")
    prev_app_smaller["AMT_APPLICATION"] = prev_app_smaller["AMT_APPLICATION"].astype("float32")
    return prev_app_smaller

class Data_processor_general():
    """
    Regroupe le traitement de toutes les tables et la préparation des données
    """

    def __init__(self, verbose=False):
        self.app_processor = Data_processor_app(verbose=verbose)
        self.prev_app_processor = Data_processor_prev_app(verbose=verbose)
        self.bureau_processor = Data_processor_bureau(verbose=verbose)

    def fit_transform(self, app_train:pd.DataFrame, prev_app:pd.DataFrame, bureau:pd.DataFrame)->pd.DataFrame:
        app_train = self.app_processor.fit_transform(app_train)
        app_train = self.prev_app_processor.fit_transform(prev_app, app_train)
        app_train = self.bureau_processor.fit_transform(bureau, app_train)
        app_train = self.optimize_dtypes(app_train, train=True)
        self.data_columns = app_train.columns
        return app_train

    def transform(self, app:pd.DataFrame, prev_app:pd.DataFrame, bureau:pd.DataFrame)->pd.DataFrame:
        app = self.app_processor.transform(app)
        app = self.prev_app_processor.transform(prev_app, app)
        app = self.bureau_processor.transform(bureau, app)
        app = self.optimize_dtypes(app, train=False)
        return app
    
    def optimize_dtypes(self, data:pd.DataFrame, train:bool=False)->pd.DataFrame:
        if train:
            # Partons du principe que l'ensemble d'entraînement est suffisamment représentatif
            # pour ce qui est du nombre de valeurs différentes prises par les features
            feature_value_counts = data.select_dtypes(["float64", "int64"]).apply(pd.Series.nunique, axis=0)
            self.binary_columns = feature_value_counts.index[feature_value_counts <= 2]
            self.float_columns = feature_value_counts.index[feature_value_counts >= 3]
        # Forcer les valeurs binaires à être des booléens
        data[self.binary_columns] = data[self.binary_columns].astype("bool")
        # Réduire la taille des valeurs décimales
        data[self.float_columns] = data[self.float_columns].astype("float32")
        return data


class Data_processor_app():
    """
    Gère la préparation des fichiers app_train.csv et app_test.csv
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def fit_transform(self, app_train:pd.DataFrame)->pd.DataFrame:
        app_train_clean = app_train.copy()
        # NB : l'odre des fonctions est important à cause de l'odre de création/modification des attributs
        app_train_clean = self.cleanup_extreme_values(app_train_clean, train=True)
        app_train_clean = self.create_new_features(app_train_clean)
        app_train_clean = self.encode_categories_preimput(app_train_clean, train=True)
        app_train_clean = self.drop_twins_preimput(app_train_clean)
        app_train_clean = self.normalize_quantities(app_train_clean, train=True)
        app_train_clean = self.impute_missing_values(app_train_clean, train=True)
        return app_train_clean
    
    def transform(self, app:pd.DataFrame)->pd.DataFrame:
        app_clean = app.copy()
        app_clean = self.cleanup_extreme_values(app_clean, train=False)
        app_clean = self.create_new_features(app_clean)
        app_clean = self.encode_categories_preimput(app_clean, train=False)
        app_clean = self.drop_twins_preimput(app_clean)
        app_clean = self.normalize_quantities(app_clean, train=False)
        app_clean = self.impute_missing_values(app_clean, train=False)
        return app_clean

    def cleanup_extreme_values(self, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Valeurs extrêmes ou indésirables dans application_train
        # CODE_GENDER : restriction aux 2 catégories dominantes ("F" (majoriatire), "M") (seulement 3 valeurs atypiques)
        app_clean.loc[~app_clean["CODE_GENDER"].isin([0, 1]), "CODE_GENDER"] = 0 # F=0
        # NAME_FAMILY_STATUS : élimination d'une valeur trop rare
        app_clean.loc[app_clean["NAME_FAMILY_STATUS"] == 5, "NAME_FAMILY_STATUS"] = 0 # Unknown=5, Married=0
        # DAY_EMPLOYED : suppression des valeurs < 0 (elles seront remplies à l'imputation)
        app_clean.loc[app_clean["DAYS_EMPLOYED"] < 0, "DAYS_EMPLOYED"] = pd.NA 
        # AMT_INCOME_TOTAL : seuillage des valeurs extrêmes
        if train:
            self.top_threshold_AMT_INCOME_TOTAL = app_clean["AMT_INCOME_TOTAL"].quantile(0.999)
        app_clean["AMT_INCOME_TOTAL"] = app_clean["AMT_INCOME_TOTAL"].clip(upper=self.top_threshold_AMT_INCOME_TOTAL)
        return app_clean
    
    def create_new_features(self, app_clean:pd.DataFrame)->pd.DataFrame:
        feat_count_before = len(app_clean.columns)
        # Création de nouvelles features à partir de application_train.csv (a besoin d'être effectué avant la normalisation)
        app_clean["RATIO_ANNUITY_CREDIT"] = app_clean["AMT_ANNUITY"] / app_clean["AMT_CREDIT"]
        # app_clean["RATIO_ANNUITY_INCOME"] = app_clean["AMT_ANNUITY"] / app_clean["AMT_INCOME_TOTAL"]
        feat_count_after = len(app_clean.columns)
        if self.verbose:
            print(f"Created {feat_count_after-feat_count_before} new features from the application table")
        return app_clean
    
    def encode_categories_preimput(self, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Encodage des catégories qui n'ont pas besoin d'imputation
        # Les catégories on déjà été encodées pour gagner de l'espace, mais ce n'est pas forcément exploitable pour de la prédiction
        # En pratique, les catégories binaires et ordinales sont déjà traitées
        # Sécurité : imputation en cas d'apparition de valeurs manquantes en test
        if train:
            self.categories = pd.Index(['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS'])
            self.category_imputer = SimpleImputer(strategy='most_frequent')
            app_clean[self.categories] = self.category_imputer.fit_transform(app_clean[self.categories])
        else:
            app_clean[self.categories] = self.category_imputer.transform(app_clean[self.categories])
        # Encodage des catégories qui ont plus de deux valeurs et ne sont pas ordinales 
        # ça ne concerne que "NAME_FAMILY_STATUS" alors on peut faire du sur-mesure
        if train:
            self.complex_categories = ["NAME_FAMILY_STATUS"]
            # groupe 1 : un peu arbitraire mais marchait bien avant; groupe 2 : choisi pour avoir du sens
            # La définition du groupe 1 mêle des choses différentes sémantiquement,
            # mais cela permet à quasiment chaque catégorie d'être encodée car une combinaison différente des deux groupes
            self.name_family_status_group_1 = {1:True, 2:True, # Célibataire (jamais marié) ou mariage civil
                                        0:False, 3:False, 4:False, 5:False} # Marié actuellement (sauf civil) ou par le passé
            self.name_family_status_group_2 = {0:True, 2:True, # Marié actuellement
                                        1:False, 3:False, 4:False, 5:False} # Pas marié actuellement
            self.encoded_categories = pd.Index(["NAME_FAMILY_STATUS_1", "NAME_FAMILY_STATUS_2"])
        app_clean["NAME_FAMILY_STATUS_1"] = app_clean["NAME_FAMILY_STATUS"].map(self.name_family_status_group_1)
        app_clean["NAME_FAMILY_STATUS_2"] = app_clean["NAME_FAMILY_STATUS"].map(self.name_family_status_group_2)
        app_clean.drop(self.complex_categories, axis=1, inplace=True)
        return app_clean
    
    def drop_twins_preimput(self, app_clean:pd.DataFrame):
        # Suppression des jumeaux (préférable de l'effectuer avant l'imputation)
        self.twins_to_drop = ["AMT_CREDIT"]
        app_clean.drop(self.twins_to_drop, axis=1, inplace=True)
        return app_clean
    
    def normalize_quantities(self, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Normalisation des quantités
        # Colonnes numériques
        if train:
            self.quantities = app_clean.columns.difference(self.categories).difference(self.encoded_categories)
        # Normalisation (centrage-réduction) des colonnes dont la distribution est suffisamment proche d'une gaussienne
        if train:
            self.quantities_normalenough = pd.Index(["RATIO_ANNUITY_CREDIT", "RATIO_ANNUITY_INCOME", "NAME_EDUCATION_TYPE", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
            self.quantities_normalenough = self.quantities_normalenough.intersection(app_clean.columns)
            self.normalenough_scaler = StandardScaler()
            app_clean[self.quantities_normalenough] = self.normalenough_scaler.fit_transform(app_clean[self.quantities_normalenough])
        else:
            app_clean[self.quantities_normalenough] = self.normalenough_scaler.transform(app_clean[self.quantities_normalenough])
        # Transformation (power transform) des colonnes dont la distribution est trop asymmétrique et étalée
        # Même si certains modèles peuvent s'accomoder des distributions non normales, ce n'est pas le cas des modèles que nous utilisons pour l'imputation
        if train:
            self.quantities_to_remap = self.quantities.difference(self.quantities_normalenough)
            self.to_remap_transformer = PowerTransformer(method='yeo-johnson', standardize=True) # yeo-johnson parce qu'on a des valeurs négatives comme positives et certaines s'annulent
            app_clean[self.quantities_to_remap] = self.to_remap_transformer.fit_transform(app_clean[self.quantities_to_remap])
        else:
            app_clean[self.quantities_to_remap] = self.to_remap_transformer.transform(app_clean[self.quantities_to_remap])
        return app_clean


    def impute_missing_values(self, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Imputation des valeurs manquantes
        # Avec les variables choisies, seules quelques variables numériques ont des valeurs manquantes

        if train:
            # Avec LightGBM, une imputation par la médiane marche mieux qu'une imputation par un modèle linéaire
            # Il faut remarquer que le R² des valeurs imputées est assez mauvais (de 0.06 à 0.12) pour la majorité des features
            # Limiter l'imputation par modèle linéaire aux seules valeurs avec un bon R² marche toujours moins bien qu'une imputation par la médiane
            # self.quantity_imputer = IterativeImputer(LinearRegression(), sample_posterior=False, initial_strategy="median", skip_complete=True, random_state=9)
            self.quantity_imputer = SimpleImputer(strategy="median")
            app_clean[app_clean.columns] = self.quantity_imputer.fit_transform(app_clean)
        else:
            app_clean[app_clean.columns] = self.quantity_imputer.transform(app_clean)
        return app_clean


class Data_processor_prev_app():
    """
    Gère la préparation de previous_application.csv, le croisement avec application_train.csv 
    et la création de features à partir de ce fichier.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def fit_transform(self, prev_app:pd.DataFrame, app_train_clean:pd.DataFrame)->pd.DataFrame:
        prev_app_clean = self.restrict_to_provided_set(prev_app, app_train_clean).copy()
        prev_app_clean = self.cleanup_extreme_values(prev_app_clean, train=True)
        prev_app_clean = self.impute_missing_values(prev_app_clean, train=True)
        merged = self.merge_tables(prev_app_clean, app_train_clean)
        app_train_clean = self.create_new_features(merged, app_train_clean, train=True)
        app_train_clean = self.normalize_quantities(app_train_clean, train=True)
        app_train_clean = self.imput_new_features(app_train_clean, train=True)
        return app_train_clean

    def transform(self, prev_app:pd.DataFrame, app_clean:pd.DataFrame)->pd.DataFrame:
        prev_app_clean = self.restrict_to_provided_set(prev_app, app_clean).copy()
        prev_app_clean = self.cleanup_extreme_values(prev_app_clean, train=False)
        prev_app_clean = self.impute_missing_values(prev_app_clean, train=False)
        merged = self.merge_tables(prev_app_clean, app_clean)
        app_clean = self.create_new_features(merged, app_clean, train=False)
        app_clean = self.normalize_quantities(app_clean, train=False)
        app_clean = self.imput_new_features(app_clean, train=False)
        return app_clean

    def restrict_to_provided_set(self, prev_app:pd.DataFrame, app_clean:pd.DataFrame)->pd.DataFrame:
        # Réduction de la table aux exemples fournis pour la rigueur et l'optimisation
        example_ids = app_clean.index # SK_ID_CURR
        prev_app_restrict = prev_app.loc[prev_app["SK_ID_CURR"].isin(example_ids)]
        # Sécurité au cas où aucun des exemples en entrée n'ait de correspondance dans prev_app
        if prev_app_restrict.shape[0] == 0:
            # La ligne de prev_app choisie importe peu vu qu'on sait qu'elle ne correspond à aucun exemple d'entrée
            # Elle sera ignorée lors de merge_tables
            # On en a quand même besoin pour éviter des plantages dans cleanup_extreme_values et imput_missing_values
            prev_app_restrict = prev_app.loc[[prev_app.index[0]]]
        return prev_app_restrict
    
    def cleanup_extreme_values(self, prev_app:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Protection générique contre les valeurs trop extrêmes
        if train:
            self.auto_quantities = prev_app.select_dtypes([float, int]).columns # inclut aussi SK_ID_CURR
            self.auto_quantities = self.auto_quantities.drop(["SK_ID_CURR"])
            self.auto_bottom_thresholds = prev_app[self.auto_quantities].quantile(0.001)
            self.auto_top_thresholds = prev_app[self.auto_quantities].quantile(0.999)
        prev_app[self.auto_quantities] = prev_app[self.auto_quantities].clip(lower=self.auto_bottom_thresholds, upper=self.auto_top_thresholds, axis=1)
        return prev_app

    def impute_missing_values(self, prev_app:pd.DataFrame, train:bool=False)->pd.DataFrame:
        #Valeurs manquantes dans prev_app
        if train:
            # Traitement générique
            # Les colonnes retenues ne devraient pas avoir de valeurs manquantes
            self.auto_quantity_imputer = SimpleImputer(strategy="median")
            prev_app[self.auto_quantities] = self.auto_quantity_imputer.fit_transform(prev_app[self.auto_quantities])
        else:
            prev_app[self.auto_quantities] = self.auto_quantity_imputer.transform(prev_app[self.auto_quantities])
        return prev_app

    def merge_tables(self, prev_app:pd.DataFrame, app_clean:pd.DataFrame)->pd.DataFrame:
        return app_clean[["CODE_GENDER"]].merge(prev_app, on="SK_ID_CURR", how='left')

    def create_new_features(self, merged:pd.DataFrame, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        self.prev_app_features = []
        # Nombre de demandes passées par demande actuelle
        # Pistes  d'amélioration : 
        # - quantifier pour les valeurs élevées
        # - séparer entre accéptées et refusées (NAME_CONTRACT_STATUS)
        no_prev_application = merged.loc[merged["SK_ID_PREV"].isna(), "SK_ID_CURR"]
        num_repeats = merged.value_counts("SK_ID_CURR", sort=False)
        num_repeats[no_prev_application] = 0
        app_clean["CNT_PREV_APP"] = num_repeats
        self.prev_app_features.append("CNT_PREV_APP")
        # Features créées automatiquement
        app_clean["PREV_APP_AMT_APPLICATION_MEAN"] =  merged["AMT_APPLICATION"].groupby(merged["SK_ID_CURR"]).aggregate('mean')
        # app_clean.loc[no_prev_application, "PREV_APP_AMT_APPLICATION_MEAN"] = #ce sera imputé plus tard
        app_clean["PREV_APP_DAYS_DECISION_STD"] = merged["DAYS_DECISION"].groupby(merged["SK_ID_CURR"]).std(ddof=0)
        app_clean.loc[no_prev_application, "PREV_APP_DAYS_DECISION_STD"] = 0.    # Il y a déjà beaucoup de valeurs à 0 à cause des groupes de 1
        self.prev_app_features.extend(["PREV_APP_AMT_APPLICATION_MEAN", "PREV_APP_DAYS_DECISION_STD"])
        if self.verbose:
            print(f"Created {len(self.prev_app_features)} new features from the previous application table")
        return app_clean

    def normalize_quantities(self, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        if train:
            self.prev_app_feat_transformer = PowerTransformer('yeo-johnson')
            app_clean[self.prev_app_features] = self.prev_app_feat_transformer.fit_transform(app_clean[self.prev_app_features])
        else:
            app_clean[self.prev_app_features] = self.prev_app_feat_transformer.transform(app_clean[self.prev_app_features])
        return app_clean
    
    def imput_new_features(self, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Imputation des valeurs manquantes pour les features créées
        if train:
            self.prev_app_feature_imputer = SimpleImputer(strategy="median")
            app_clean[self.prev_app_features] = self.prev_app_feature_imputer.fit_transform(app_clean[self.prev_app_features])
        else:
            app_clean[self.prev_app_features] = self.prev_app_feature_imputer.transform(app_clean[self.prev_app_features])
        return app_clean


class Data_processor_bureau():
    """
    Gère la préparation de previous_application.csv, le croisement avec application_train.csv 
    et la création de features à partir de ce fichier.
    """
    # Colonnes qui font l'objet d'un traitement particulier
    focus_cols = ["CREDIT_ACTIVE", "AMT_CREDIT_MAX_OVERDUE"]

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit_transform(self, bureau:pd.DataFrame, app_train_clean:pd.DataFrame)->pd.DataFrame:
        bureau_clean = self.restrict_to_provided_set(bureau, app_train_clean).copy()
        bureau_clean = self.cleanup_extreme_values(bureau_clean, train=True)
        bureau_clean = self.imput_missing_values(bureau_clean, train=True)
        merged = self.merge_tables(bureau_clean, app_train_clean)
        app_train_clean = self.create_new_features(merged, app_train_clean, train=True)
        app_train_clean = self.normalize_quantities(app_train_clean, train=True)
        app_train_clean = self.imput_new_features(app_train_clean, train=True)
        return app_train_clean
    
    def transform(self, bureau:pd.DataFrame, app_clean:pd.DataFrame)->pd.DataFrame:
        bureau_clean = self.restrict_to_provided_set(bureau, app_clean).copy()
        bureau_clean = self.cleanup_extreme_values(bureau_clean, train=False)
        bureau_clean = self.imput_missing_values(bureau_clean, train=False)
        merged = self.merge_tables(bureau_clean, app_clean)
        app_clean = self.create_new_features(merged, app_clean, train=False)
        app_clean = self.normalize_quantities(app_clean, train=False)
        app_clean = self.imput_new_features(app_clean, train=False)
        return app_clean
    
    def restrict_to_provided_set(self, bureau:pd.DataFrame, app_clean:pd.DataFrame)->pd.DataFrame:
        # Réduction de la table aux exemples fournis pour la rigueur et l'optimisation
        example_ids = app_clean.index # SK_ID_CURR
        bureau_restrict = bureau.loc[bureau["SK_ID_CURR"].isin(example_ids)]
        # Sécurité au cas où aucun des exemples en entrée n'ait de correspondance dans bureau
        if bureau_restrict.shape[0] == 0:
            # La ligne de bureau choisie importe peu vu qu'on sait qu'elle ne correspond à aucun exemple d'entrée
            # Elle sera ignorée lors de merge_tables
            # On en a quand même besoin pour éviter des plantages dans cleanup_extreme_values et imput_missing_values
            bureau_restrict = bureau.loc[[bureau.index[0]]]
        return bureau_restrict

    def cleanup_extreme_values(self, bureau_clean:pd.DataFrame, train:bool=False):
        # Valeurs extrêmes ou indésirables dans bureau
        # CREDIT_ACTIVE : restriction aux 2 catégories dominantes
        ok_values_CREDIT_ACTIVE = [0, 1] # ["Closed", "Active"]
        cond = ~bureau_clean["CREDIT_ACTIVE"].isin(ok_values_CREDIT_ACTIVE)
        bureau_clean.loc[cond, "CREDIT_ACTIVE"] = 1 # "Active"
        # AMT_CREDIT_MAX_OVERDUE : seuillage des valeurs extrêmes
        credit_max_overdue = bureau_clean["AMT_CREDIT_MAX_OVERDUE"]
        if train:
            non_null = credit_max_overdue[credit_max_overdue > 0]
            self.top_threshold_AMT_CREDIT_MAX_OVERDUE = non_null.quantile(0.995)
        bureau_clean["AMT_CREDIT_MAX_OVERDUE"] = bureau_clean["AMT_CREDIT_MAX_OVERDUE"].clip(upper=self.top_threshold_AMT_CREDIT_MAX_OVERDUE)
        # Protection générique contre les valeurs trop extrêmes pour les autres quantités
        if train:
            self.auto_quantities = bureau_clean.select_dtypes([float, int]).columns
            self.auto_quantities = self.auto_quantities.drop(["SK_ID_CURR"])
            self.auto_quantities_to_clip = self.auto_quantities.difference(pd.Index(self.focus_cols))
            self.auto_bottom_thresholds = bureau_clean[self.auto_quantities_to_clip].quantile(0.001)
            self.auto_top_thresholds = bureau_clean[self.auto_quantities_to_clip].quantile(0.999)
        bureau_clean[self.auto_quantities_to_clip] = bureau_clean[self.auto_quantities_to_clip].clip(lower=self.auto_bottom_thresholds, upper=self.auto_top_thresholds, axis=1)
        return bureau_clean

    def imput_missing_values(self, bureau_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Valeurs manquantes dans bureau
        miss_val_AMT_CREDIT_MAX_OVERDUE = bureau_clean["AMT_CREDIT_MAX_OVERDUE"].isna()
        bureau_clean.loc[miss_val_AMT_CREDIT_MAX_OVERDUE, "AMT_CREDIT_MAX_OVERDUE"] = 0
        # Traitement générique pour les autres quantités
        if train:
            self.auto_quantity_imputer = SimpleImputer(strategy="median")
            bureau_clean[self.auto_quantities] = self.auto_quantity_imputer.fit_transform(bureau_clean[self.auto_quantities])
        else:
            bureau_clean[self.auto_quantities] = self.auto_quantity_imputer.transform(bureau_clean[self.auto_quantities])
        return bureau_clean
    
    def merge_tables(self, bureau_clean:pd.DataFrame, app_clean:pd.DataFrame)->pd.DataFrame:
        return app_clean[["CODE_GENDER"]].merge(bureau_clean, on="SK_ID_CURR", how='left')

    def create_new_features(self, merged:pd.DataFrame, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Création de features à partir de bureau.csv
        self.bureau_features = []
        # Nombre de crédits chez d'autres institutions par demande
        no_other_credit = merged.loc[merged["SK_ID_BUREAU"].isna(), "SK_ID_CURR"]
        num_repeats = merged.value_counts("SK_ID_CURR", sort=False)
        num_repeats[no_other_credit] = 0
        app_clean["CNT_OTHER_CREDIT"] = num_repeats
        self.bureau_features.append("CNT_OTHER_CREDIT")
        # Nombre de crédits actifs chez d'autres institutions par demande (trop corrélé à CNT_OTHER_CREDIT)
        cnt_other_credit_active = (merged["CREDIT_ACTIVE"] == 1).groupby(merged["SK_ID_CURR"]).aggregate('sum') 
        # Proportion de crédits actifs par rapport à l'historique des crédits
        app_clean["RATIO_CREDIT_ACTIVE"] = cnt_other_credit_active / app_clean["CNT_OTHER_CREDIT"]
        app_clean.loc[no_other_credit, "RATIO_CREDIT_ACTIVE"] = 0.
        self.bureau_features.append("RATIO_CREDIT_ACTIVE")
        # Montant maximal qui a été en retard 
        app_clean["AMT_CREDIT_MAX_OVERDUE"] = merged["AMT_CREDIT_MAX_OVERDUE"].groupby(merged["SK_ID_CURR"]).aggregate("max")
        app_clean.loc[app_clean["AMT_CREDIT_MAX_OVERDUE"].isna(), "AMT_CREDIT_MAX_OVERDUE"] = 0
        self.bureau_features.append("AMT_CREDIT_MAX_OVERDUE")
        # Features créées automatiquement
        app_clean['BUREAU_AMT_CREDIT_SUM_MEAN'] = merged["AMT_CREDIT_SUM"].groupby(merged["SK_ID_CURR"]).aggregate('mean')
        app_clean['BUREAU_DAYS_CREDIT_MEAN'] = merged["DAYS_CREDIT"].groupby(merged["SK_ID_CURR"]).aggregate('mean')
        # les valeurs manquantes seront imputées plus tard
        self.bureau_features.extend(['BUREAU_AMT_CREDIT_SUM_MEAN', 'BUREAU_DAYS_CREDIT_MEAN'])
        if self.verbose:
            print(f"Created {len(self.bureau_features)} new features from the bureau table")
        return app_clean

    def normalize_quantities(self, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Normalisation
        if train:
            self.bureau_quantities = app_clean[self.bureau_features].select_dtypes([float, int]).columns
            self.bureau_quantities_ratios = ["RATIO_CREDIT_ACTIVE"] # Ce(s) feature(s) sont déjà restreintes à [0,1] avec beaucoup de valeurs à 0 et à 1
            self.bureau_quantities = self.bureau_quantities.difference(self.bureau_quantities_ratios)
            self.bureau_quantity_transformer = PowerTransformer(method='yeo-johnson')
            app_clean[self.bureau_quantities] = self.bureau_quantity_transformer.fit_transform(app_clean[self.bureau_quantities])
        else:
            app_clean[self.bureau_quantities] = self.bureau_quantity_transformer.transform(app_clean[self.bureau_quantities])
        return app_clean
    
    def imput_new_features(self, app_clean:pd.DataFrame, train:bool=False)->pd.DataFrame:
        # Imputation des valeurs manquantes pour les features créées
        if train:
            self.bureau_feature_imputer = SimpleImputer(strategy="median")
            app_clean[self.bureau_features] = self.bureau_feature_imputer.fit_transform(app_clean[self.bureau_features])
        else:
            app_clean[self.bureau_features] = self.bureau_feature_imputer.transform(app_clean[self.bureau_features])
        return app_clean

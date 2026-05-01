import hashlib
from base64 import urlsafe_b64encode
import inspect
import os
import shutil
import pickle
import pandas as pd

# hachage pour identifier des données dans un nom de fichier
def b64hash(data)->str:
    md5 = hashlib.md5(data)
    return urlsafe_b64encode(md5.digest()).decode()[:-2]

class Caching_processor():
    """
    Encapsule un préparateur de données en ajoutant un comportement de mise en cache des préparateurs entraînés et des données.

    En production, ses fonctionnalités spécifiques doivent être désactivées et il reste principalement en tant que vestige du processus d'entraînement.
    """

    def __init__(self, generated_dir:os.PathLike, prev_app_path:os.PathLike, bureau_path:os.PathLike, Processor_class:type):
        """
        Paramètres :
        - generated_dir : dossier où ranger les préparateurs mis en cache
        - les tables à croiser avec la table principale pour la préparation des données
        - processor_class : paramètre expérimental pour changer le préparateur de données
        """
        self.generated_dir = generated_dir
        self.prev_app_path = prev_app_path
        self.bureau_path = bureau_path
        self.prev_app = None
        self.bureau = None
        self.Processor_class = Processor_class
        self.caching_enabled = True
        self.mlflow_mode = False
    
    def __getstate__(self):
        # Modifie le comportement pour éviter de pickler des tas de copies des tables secondaires
        self.prev_app = None
        self.bureau = None
        return self.__dict__
    
    def disable_caching(self):
        """
        Désactive la mise en cache des exemples.
        À utiliser avant le déploiement du modèle, car on ne voit jamais le même exemple deux fois de suite en production.

        Désactive aussi les affichages de débuggage dans transform
        """
        self.caching_enabled = False

    def enable_mlflow_mode(self):
        """
        Change le comportement du modèle pour s'adapter au bug de mlflow qui jette l'index des Dataframes
        """
        self.mlflow_mode = True
    
    def reset_paths(self, generated_dir:os.PathLike, prev_app_path:os.PathLike, bureau_path:os.PathLike):
        """
        Change les chemins de fichiers sauvegardés par l'objet, pour une utilisation dans un autre contexte
        """
        self.generated_dir = generated_dir
        self.prev_app_path = prev_app_path
        self.bureau_path = bureau_path
    
    def fit_transform(self, app_train:pd.DataFrame, y=None):
        """
        Prépare les données, en entraînant le pipeline de traitement des données si nécessaire ou bien en chargeant une version déjà traitée sinon.

        Le pipeline doit être exécuté si son code a chagé ou si les données n'ont jamais été préparées par ce pipeline.
        Si c'est le cas, le pipeline entraîné et les données préparées seront sauvegardées pour une utilisation future.
        """
        # Vérification si le pipeline de préparation des données ou les données ont changé
        fe_module_path = inspect.getabsfile(self.Processor_class)
        with open(fe_module_path, "rt") as f:
            lines = f.readlines()
        code_hash = b64hash(bytes("".join(lines), encoding="utf-8"))
        processed_data_dir = os.path.join(self.generated_dir, "fe_"+code_hash)
        new_data_processor = False
        if not os.path.exists(processed_data_dir):
            new_data_processor = True
        train_data_hash = b64hash(pd.util.hash_pandas_object(app_train).values)
        train_data_path = os.path.join(processed_data_dir, f"app_{train_data_hash}.parquet")
        data_processor_path = os.path.join(processed_data_dir, f"data_processor_{train_data_hash}.pickle")
        new_train_data = False
        if not os.path.exists(train_data_path):
            new_train_data = True
        # Préparation des données
        if new_data_processor or new_train_data:
            print("Entraînement d'un nouveau préparateur")
            print(f"Préparateur : {code_hash} ; Données : {train_data_hash}")
            data_processor = self.Processor_class()
            if self.prev_app is None:
                self._load_secondary_tables()
            app_train = data_processor.fit_transform(app_train, self.prev_app, self.bureau)
            # Mise en cache de tout ce qui est pertinent
            if self.caching_enabled:
                if new_data_processor:
                    os.mkdir(processed_data_dir)
                    shutil.copy(fe_module_path, processed_data_dir)
                with open(data_processor_path, "wb") as f:
                    pickle.dump(data_processor, f, 5)
                app_train.to_parquet(train_data_path, index=True)
        # Chargement depuis le cache
        else:
            with open(data_processor_path, "rb") as f:
                data_processor = pickle.load(f)
            app_train = pd.read_parquet(train_data_path)
        self.code_hash = code_hash
        self.processed_data_dir = processed_data_dir
        self.data_processor = data_processor
        return app_train

    def transform(self, app_test:pd.DataFrame):
        """
        Prépare les données, en éxécutant le pipeline de traitement des données si nécessaire ou bien en chargeant une version déjà traitée sinon.

        Le pipeline doit être exécuté si les données n'ont jamais été préparées par ce pipeline.
        Si c'est le cas, le pipeline entraîné et les données préparées seront sauvegardées pour une utilisation future.
        """
        # Vérification si les données sont nouvelles
        if self.caching_enabled:
            test_data_hash = b64hash(pd.util.hash_pandas_object(app_test).values)
            test_data_path = os.path.join(self.processed_data_dir, f"app_{test_data_hash}.parquet")
            new_test_data = False
            if not os.path.exists(test_data_path):
                new_test_data = True
        else:
            new_test_data = True

        if self.mlflow_mode:
            app_test = app_test.set_index("SK_ID_CURR")

        # Préparation des données
        if new_test_data:
            if self.caching_enabled:
                print("Préparation de nouvelles données")
                print(f"Préparateur : {self.code_hash} ; Données : {test_data_hash}")
            if self.prev_app is None:
                self._load_secondary_tables()
            app_test = self.data_processor.transform(app_test, self.prev_app, self.bureau)
            # Mise en cache de tout ce qui est est pertinent
            if self.caching_enabled:
                app_test.to_parquet(test_data_path, index=True)
        # Chargement depuis le cache
        else:
            app_test = pd.read_parquet(test_data_path)
        return app_test
    
    def _load_secondary_tables(self):
        def read_csv_or_parquet(path):
            root, filext = os.path.splitext(path)
            if filext == ".csv":
                table = pd.read_csv(path)
            elif filext == ".parquet":
                table = pd.read_parquet(path)
            return table
        self.prev_app = read_csv_or_parquet(self.prev_app_path)
        self.bureau = read_csv_or_parquet(self.bureau_path)
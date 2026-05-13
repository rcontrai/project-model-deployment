# Exporte les logs en .parquet à raison d'un fichier par table de logging
# Utilisation : uv run -m logs.export_logs

from sqlmodel import create_engine, Session, select
import pandas as pd
import huggingface_hub
import os
from src.api import log_tables, HF_BUCKET_URL, HF_BUCKET_TOKEN

LOGS_DIR = "./logs/"

# Si on est dans un environnement qui a accès au bucket, télécharger les logs du bucket
if HF_BUCKET_TOKEN is not None:
    huggingface_hub.login(HF_BUCKET_TOKEN)
    logfiles = [
        item for item in huggingface_hub.list_bucket_tree(HF_BUCKET_URL, prefix="logs/")
        if item.type == "file" and item.path.endswith(".db")
    ]
    huggingface_hub.download_bucket_files(
        HF_BUCKET_URL,
        files=[(file, os.path.join(LOGS_DIR, os.path.basename(file.path))) for file in logfiles]
    )
    huggingface_hub.logout()

# Lire tous les .db du dossier
files = os.listdir(LOGS_DIR)
logfiles = [os.path.join(LOGS_DIR, file) for file in files if file.endswith(".db")]
table_exports = {table.__name__:[] for table in log_tables}
for logfile in logfiles:
    engine = create_engine(f"sqlite:///{logfile}")
    for table in log_tables:
        rows = []
        with Session(engine) as session:
            results = session.exec(select(table))
            for log in results:
                rows.append(log.model_dump())
        if len(rows) > 0:
            df = pd.DataFrame.from_records(rows)
            df["logfile"] = os.path.basename(logfile)
            table_exports[table.__name__].append(df)

# Exporter sous forme de .parquet
for table in log_tables:
    table_name = table.__name__
    output_path = os.path.join(LOGS_DIR, f"log_{table_name}.parquet")
    dataframes:list[pd.DataFrame] = table_exports[table_name]
    if len(dataframes) > 0:
        df = pd.concat(dataframes, axis=0, ignore_index=True)
        df.to_parquet(output_path)
    else :
        print(f"Nothing to export for table {table_name} !")
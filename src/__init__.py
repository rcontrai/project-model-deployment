# Ajout du dossier src au path parce que le modèle picklé contient des références à des modules à la racine du path
from pathlib import Path
import sys
_base_dir = str(Path(__file__).parent)
if _base_dir not in sys.path:
    sys.path.insert(1, _base_dir)
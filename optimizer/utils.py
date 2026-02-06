import json
from pathlib import Path
from typing import Any, Dict


def load_input_data(path: str | Path) -> Dict[str, Any]:

    #Load and parse JSON input data from the given path.

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data

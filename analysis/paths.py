from pathlib import Path


class Folders:
    home: Path = Path.home()
    root: Path = Path.cwd()
    analysis: Path = root / "analysis"

    output: Path = analysis / "output"
    plots: Path = analysis / "plots"
    statistics: Path = analysis / "statistics"
    assets: Path = analysis / "assets"
    data: Path = analysis / "data"

    attributions: Path = data / "attributions_npy"
    predictions: Path = data / "predictions"


class Files:
    parameters: Path = Folders.assets / "parameters.json"
    variables: Path = Folders.assets / "variables.json"

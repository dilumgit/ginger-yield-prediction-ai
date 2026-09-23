"""Data loader module for raw research dataset.

Provides pure, non-destructive loading functions for raw ginger dataset files.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.config import RAW_DATA_PATH


def load_raw_data(
    filepath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the raw ginger dataset from disk.

    Parameters
    ----------
    filepath : str or Path, optional
        Custom path to the CSV file. If None, defaults to `RAW_DATA_PATH`
        configured in `src.utils.config`.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified DataFrame containing all rows and columns.

    Raises
    ------
    FileNotFoundError
        If the raw dataset CSV is not found at the specified location.
    ValueError
        If the file is empty or cannot be parsed as a valid CSV.
    """
    path = Path(filepath) if filepath is not None else RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at '{path.resolve()}'.\n"
            f"Please place the raw ginger research dataset at:\n"
            f"  {RAW_DATA_PATH.resolve()}\n"
            f"Note: Do not generate synthetic data or alter column headers."
        )

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(
            f"Failed to read CSV dataset from '{path.resolve()}': {e}"
        ) from e

    if df.empty:
        raise ValueError(f"Dataset at '{path.resolve()}' is empty.")

    # Return a defensive copy to prevent accidental in-place mutations
    return df.copy()

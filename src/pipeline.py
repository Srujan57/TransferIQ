"""
Quick-run pipeline. Execute from project root:
    python -m src.pipeline
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.data_loader import load_data, load_training_data
from src.integration.final_equation import TransferIQValuation


def main():
    df_train = load_training_data(season_cutoff=2023)
    df_test  = load_data(test_season_cutoff=2024)
    engine = TransferIQValuation()
    engine.train(df_train, df_test)
    engine.save('models/transferiq_model.pkl')
    print("\nPipeline complete.")


if __name__ == '__main__':
    main()

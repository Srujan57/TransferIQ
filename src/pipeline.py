"""
Quick-run pipeline. Execute from project root:
    python -m src.pipeline
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.data_loader import load_data
from src.integration.final_equation import TransferIQValuation


def main():
    df = load_data()
    engine = TransferIQValuation()
    engine.train(df)
    engine.save('models/transferiq_model.pkl')
    print("\nPipeline complete.")


if __name__ == '__main__':
    main()

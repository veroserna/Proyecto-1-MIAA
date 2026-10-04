"""Punto de entrada principal del proyecto."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import load_data, clean_data
from src.models.my_model import MyModel
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model
from src.utils.helpers import print_header


def main():
    print_header("Proyecto MIAA")
    data_path = os.path.join(PROJECT_ROOT, "sources", "AmesProperty.csv")
    data = load_data(data_path)
    cleaned = clean_data(data)
    model = MyModel(name="modelo_base")
    training_result = train_model(model, cleaned)
    evaluation_result = evaluate_model(model, [1, 2, 3, 4])
    print(training_result)
    print(evaluation_result)


if __name__ == "__main__":
    main()

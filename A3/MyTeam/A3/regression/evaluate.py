import os
import torch
import argparse
import numpy as np
import pandas as pd

from models import Random_Regressor, Linear_Regressor, Custom_Regressor, load_data


def tocsv(y_arr, *, task):
    r"""Writes the numpy array to a csv file.
    params:
        y_arr: np.ndarray. A vector of all the predictions. Classes for
        classification and the regression value predicted for regression.

        task: str. Must be either of "classification" or "regression".
        Must be a keyword argument.
    Outputs a file named "y_classification.csv" or "y_regression.csv" in
    the directory it is called from. Must only be run once. In case outputs
    are generated from batches, only call this output on all the predictions
    from all the batches collected in a single numpy array. This means it'll
    only be called once.

    This code ensures this by checking if the file already exists, and does
    not over-write the csv files. It just raises an error.

    Finally, do not shuffle the test dataset as then matching the outputs
    will not work.
    """
    assert task in ["classification",
                    "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(
        y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze(
    ).shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(
        f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)


def predict(model_path, test_data_path, model='custom'):
    SHOW_ACCURACY = False
    dataset, dataloader = load_data(
        test_data_path, batch_size=-1, load_labels=SHOW_ACCURACY, shuffle=False)

    if model == 'random':
        MODEL = Random_Regressor(num_classes=2)
    elif model == 'custom':
        MODEL = Custom_Regressor(
            in_channels=dataset.num_features, hidden_channels=32, out_channels=1)
        MODEL.load_state_dict(torch.load(model_path))
    else:
        MODEL = Linear_Regressor(
            in_channels=dataset.num_features, hidden_channels=32, out_channels=1)
        MODEL.load_state_dict(torch.load(model_path))

    data = next(iter(dataloader))
    output = MODEL.predict(data)

    if SHOW_ACCURACY:
        val_loss = torch.nn.MSELoss()(output, data.y)
        print(f"Loss with {model} model: {val_loss:.4f}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluating the regression model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model", required=False, default='custom', type=str)
    args = parser.parse_args()
    print(
        f"Evaluating the regression model. Model will be loaded from {args.model_path}. Test dataset will be loaded from {args.dataset_path}.")

    predicted_output = predict(args.model_path, args.dataset_path, args.model)
    output = predicted_output.cpu().numpy()
    tocsv(output, task="regression")


if __name__ == "__main__":
    main()

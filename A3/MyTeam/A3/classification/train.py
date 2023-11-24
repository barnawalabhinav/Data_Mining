import torch
import argparse

from models import Custom_Classifier, Logistic_Regressor, load_data

def train(model_path, train_data_path, val_data_path, num_epochs=200, batch_size=32, model='custom', checkpoint_path=None):
    dataset, dataloader = load_data(train_data_path, batch_size)
    _, val_dataloader = load_data(val_data_path, -1)
    val_data = next(iter(val_dataloader))

    if model == 'custom':
        MODEL = Custom_Classifier(
            in_channels=dataset.num_features, out_channels=1, edge_dim=dataset.num_edge_features)
    else:
        print(f'Running the baseline model')
        MODEL = Logistic_Regressor(
            in_channels=dataset.num_features, out_channels=1)

    if checkpoint_path is not None:
        MODEL.load_state_dict(torch.load(checkpoint_path))

    # optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=1e-3)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(MODEL.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    best_loss = float('inf')
    best_acc = 0
    for epoch in range(num_epochs):
        train_loss = 0
        correct_output = 0
        num_batches = 0
        num_graphs = 0
        for batch in dataloader:
            num_graphs += batch.num_graphs
            optimizer.zero_grad()
            output = MODEL(batch)
            labels = torch.where(output < 0.5, torch.tensor(0.0), torch.tensor(1.0))
            # gold = F.one_hot(batch.y, num_classes=2).to(dtype=torch.float)
            loss = criterion(output, batch.y)

            # ******************* For SVM *******************
            # Add regularization i.e.  Full loss = data loss + regularization loss
            weight = MODEL.classifier.weight.squeeze()
            loss += 0.01 * torch.sum(weight * weight)
            # loss += 0.01 * torch.sum(1.0 * weight * weight + torch.abs(weight))
            # -----------------------------------------------

            loss.backward()
            optimizer.step()

            train_loss += loss
            num_batches += 1
            correct_output += torch.sum(batch.y == labels).item()

        if epoch % 1 == 0:
            output = MODEL.predict(val_data)
            val_loss = criterion(output, val_data.y)
            labels = torch.where(output < 0.5, torch.tensor(0.0), torch.tensor(1.0))
            correct_val = torch.sum(val_data.y == labels).item()
            val_acc = correct_val / val_data.num_graphs * 100
            print('-------------------------------------------')
            print(f'Epoch: {epoch:03d}')
            print(f'Train Loss: {train_loss/num_batches:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Train Accuracy: {correct_output / num_graphs * 100 :.2f} %')
            print(f'Val Accuracy: {val_acc :.2f} %')
            # if val_loss < best_loss:
                # best_loss = val_loss
            if val_acc > best_acc:
                best_acc = val_acc
                print('Saving the best model')
                torch.save(MODEL.state_dict(), model_path)

    # torch.save(MODEL.state_dict(), model_path)


def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--checkpoint", required=False, default=None, type=str)
    parser.add_argument("--num_epochs", required=False, default=200, type=int)
    parser.add_argument("--batch_size", required=False, default=32, type=int)
    parser.add_argument("--model", required=False, default='custom', type=str)
    args = parser.parse_args()
    print(
        f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")

    train(args.model_path, args.dataset_path, args.val_dataset_path,
          args.num_epochs, args.batch_size, args.model, args.checkpoint)


if __name__ == "__main__":
    main()

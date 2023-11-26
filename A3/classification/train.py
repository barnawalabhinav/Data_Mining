import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from models import Custom_Classifier, Logistic_Regressor, load_data, hinge_loss, svm_loss


def train(model_path, train_data_path, val_data_path, num_epochs=200, batch_size=32, model='custom', checkpoint_path=None, plot_path=None):
    dataset, dataloader = load_data(train_data_path, batch_size)
    _, val_dataloader = load_data(val_data_path, -1)
    val_data = next(iter(val_dataloader))

    if model == 'custom':
        MODEL = Custom_Classifier(
            in_channels=dataset.num_features, out_channels=1, edge_dim=dataset.num_edge_features)
    else:
        print(f'Running the baseline model')
        MODEL = Logistic_Regressor(in_channels=dataset.num_features, out_channels=1)

    if checkpoint_path is not None:
        MODEL.load_state_dict(torch.load(checkpoint_path))

    MODEL.train()
    # optimizer = torch.optim.Adam(class_model.parameters(), lr=0.01, weight_decay=1e-3)
    # optimizer = torch.optim.Adam(MODEL.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    if plot_path is not None:
        epoch_num = []
        val_losses = []
        train_losses = []

    grad_norms = []

    best_score = 0
    for epoch in range(num_epochs):
        mean_norm = 0
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

            # # # ****************** Custom loss functions ******************
            # # # Convert labels to -1 and 1
            # # labels[labels == 0] = -1
            # output_labels = torch.where(output < 0.5, torch.tensor(-1.0), torch.tensor(1.0))
            # true_labels = torch.where(batch.y < 0.5, torch.tensor(-1.0), torch.tensor(1.0))

            # # # Get the weights and bias from your model
            # weights = MODEL.classifier.weight
            # bias = MODEL.classifier.bias

            # # # Compute the loss
            # # loss = svm_loss(output_labels, true_labels, weights, bias, C=1.0)
            # SVMLoss = torch.nn.HingeEmbeddingLoss()
            # loss = SVMLoss(output_labels, true_labels)
            # # # -----------------------------------------------

            loss.backward()
            optimizer.step()

            # ************** Gradient Clipping **************
            norm = torch.norm(torch.cat([p.grad.flatten() for p in MODEL.parameters() if p.grad is not None]))
            mean_norm += norm

            train_output = MODEL.predict(batch)
            train_loss += criterion(train_output, batch.y)
            num_batches += 1
            correct_output += torch.sum(batch.y == labels).item()
        
        grad_norms.append(mean_norm / num_batches)

        if epoch % 1 == 0:
            output = MODEL.predict(val_data)
            val_loss = criterion(output, val_data.y)
            labels = torch.where(output < 0.5, torch.tensor(0.0), torch.tensor(1.0))
            correct_val = torch.sum(val_data.y == labels).item()
            val_acc = correct_val / val_data.num_graphs * 100
            val_score = roc_auc_score(val_data.y.detach().cpu().numpy(), output.detach().cpu().numpy())

            print('-------------------------------------------')
            print(f'Epoch: {epoch:03d}')
            print(f'Train Loss: {train_loss/num_batches:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Train Accuracy: {correct_output / num_graphs * 100 :.2f} %')
            print(f'Val Accuracy: {val_acc :.2f} %')
            print(f'Val ROC_AUC_score: {val_score :.4f}')

            if val_score > best_score:
                best_score = val_score
                print('Saving the best model')
                torch.save(MODEL.state_dict(), model_path)

            if plot_path is not None:
                epoch_num.append(epoch)
                val_losses.append(val_loss)
                train_losses.append(train_loss/num_batches)

    if plot_path is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epoch_num, val_losses, label='Val Loss')
        ax.plot(epoch_num, train_losses, label='Train Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('BCE Loss')
        if model == 'custom':
            ax.set_title('Custom Classifier')
        else:
            ax.set_title('Baseline Classifier')
        ax.legend()
        plt.savefig('loss_' + plot_path)

        plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epoch_num, grad_norms, label='Gradient Norm')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Gradient Norm')
        if model == 'custom':
            ax.set_title('Custom Classifier')
        else:
            ax.set_title('Baseline Classifier')
        ax.legend()
        plt.savefig('grad_' + plot_path)


def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--checkpoint", required=False, default=None, type=str)
    parser.add_argument("--num_epochs", required=False, default=300, type=int)
    parser.add_argument("--batch_size", required=False, default=32, type=int)
    parser.add_argument("--model", required=False, default='custom', type=str)
    parser.add_argument("--plot_path", required=False, default=None, type=str)
    args = parser.parse_args()
    print(
        f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")

    train(args.model_path, args.dataset_path, args.val_dataset_path,
          args.num_epochs, args.batch_size, args.model, args.checkpoint, args.plot_path)


if __name__ == "__main__":
    main()

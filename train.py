import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, batch_size=64):
    """
    Trains the model
    """
    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Prepare DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)

        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_func(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = loss_func(outputs, batch_y)
                val_loss += loss.item()

                predictions = torch.sigmoid(outputs) >= 0.5
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct / total) * 100

        print(f" Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Val Acc: {val_accuracy:.2f}%")

    print("Training complete.")


def test_model(model, X_test, y_test, batch_size=64):
    """
    Evaluates the model
    """
    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    model.eval()
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)

            predictions = (torch.sigmoid(outputs) >= 0.5).float()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['<=50K', '>50K'])
    matrix = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 30)
    print("FINAL TEST RESULTS")
    print("=" * 30)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)

    return acc
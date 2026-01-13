import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_model(model, X_train, y_train, epochs=100, lr=0.001):
    """
    Trains the model
    """
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)

        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = loss_func(outputs, batch_y)
            loss.backward()

            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss

            progress_bar.set_postfix(loss=f"{current_loss:.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] Completed - Avg Loss: {avg_loss:.4f}")

    print("Training complete.")
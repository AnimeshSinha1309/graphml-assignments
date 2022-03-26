import torch
import tqdm.auto as tqdm


def training(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss()
    training_losses, validation_losses, training_accuracy, validation_accuracy = (
        [],
        [],
        [],
        [],
    )

    with tqdm.trange(1, 200 + 1) as iterator:
        iterator.set_description("Training")

        for _epoch in iterator:
            model.train()
            optimizer.zero_grad()
            predictions = model(data.x, data.edge_index)
            accuracy = torch.mean(
                torch.eq(
                    torch.argmax(predictions[data.train_mask], dim=1),
                    data.y[data.train_mask],
                ).float()
            )
            loss = loss_fn(predictions[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            predictions = model(data.x, data.edge_index)
            val_loss = loss_fn(predictions[data.val_mask], data.y[data.val_mask])
            val_accuracy = torch.mean(
                torch.eq(
                    torch.argmax(predictions[data.val_mask], dim=1),
                    data.y[data.val_mask],
                ).float()
            )
            optimizer.step()

            iterator.set_postfix(
                training_loss=loss.item(),
                validation_loss=val_loss.item(),
                training_accuracy=accuracy.item(),
                val_accuracy=val_accuracy.item(),
            )
            training_losses.append(loss.item())
            validation_losses.append(val_loss.item())

    return training_losses, validation_losses, training_accuracy, validation_accuracy

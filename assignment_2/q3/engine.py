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

            with torch.no_grad():
                model.eval()
                predictions = model(data.x, data.edge_index)
                val_loss = loss_fn(predictions[data.val_mask], data.y[data.val_mask])
                val_accuracy = torch.mean(
                    torch.eq(
                        torch.argmax(predictions[data.val_mask], dim=1),
                        data.y[data.val_mask],
                    ).float()
                )

            iterator.set_postfix(
                train_loss=loss.item(),
                val_loss=val_loss.item(),
                train_acc=accuracy.item(),
                val_acc=val_accuracy.item(),
            )
            training_losses.append(loss.item())
            validation_losses.append(val_loss.item())
            training_accuracy.append(accuracy.item())
            validation_accuracy.append(val_accuracy.item())

    return training_losses, validation_losses, training_accuracy, validation_accuracy


def rnn_train(model):
    from tensorflow.keras.datasets import imdb

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=1000)

    word2id = imdb.get_word_index()
    id2word = {i: word for word, i in word2id.items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss()
    training_losses, validation_losses, training_accuracy, validation_accuracy = (
        [],
        [],
        [],
        [],
    )

    for epoch in range(5):
        total_accuracy = 0
        total_loss = 0
        total_val_accuracy = 0
        total_val_loss = 0

        with tqdm.tqdm(list(zip(X_train, y_train))) as iterator:
            iterator.set_description(f"Training Epoch {epoch}")
            counter = 0
            for x, y in iterator:
                model.train()
                optimizer.zero_grad()

                x_onehot = torch.zeros((len(x), 1000)).scatter_(
                    1,
                    torch.tensor(
                        [
                            [
                                token,
                            ]
                            for token in x
                        ]
                    ).long(),
                    1,
                )
                y = torch.tensor(y).long()
                predictions = model(x_onehot)
                accuracy = torch.mean(
                    torch.eq(
                        torch.argmax(predictions, dim=0),
                        y,
                    ).float()
                )
                loss = loss_fn(predictions, y)
                loss.backward()
                optimizer.step()

                counter += 1
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                iterator.set_postfix(
                    train_loss=total_loss / counter,
                    train_acc=total_accuracy / counter,
                )

        with torch.no_grad():
            model.eval()
            with tqdm.tqdm(list(zip(X_test, y_test))) as iterator:
                iterator.set_description(f"Testing Epoch {epoch}")
                counter = 0
                for x, y in iterator:
                    x_onehot = torch.zeros((len(x), 1000)).scatter_(
                        1,
                        torch.tensor(
                            [
                                [
                                    token,
                                ]
                                for token in x
                            ]
                        ).long(),
                        1,
                    )
                    y = torch.tensor(y).long()
                    predictions = model(x_onehot)
                    val_accuracy = torch.mean(
                        torch.eq(
                            torch.argmax(predictions, dim=0),
                            y,
                        ).float()
                    )
                    val_loss = loss_fn(predictions, y)

                    counter += 1
                    total_val_loss += val_loss.item()
                    total_val_accuracy += val_accuracy.item()
                    iterator.set_postfix(
                        val_loss=total_val_loss / counter,
                        val_acc=total_val_accuracy / counter,
                    )

        training_losses.append(total_loss / len(X_train))
        validation_losses.append(total_val_loss / len(X_test))
        training_accuracy.append(total_accuracy / len(X_train))
        validation_accuracy.append(total_val_accuracy / len(X_test))

    return training_losses, validation_losses, training_accuracy, validation_accuracy


# %% -------------------------------------------ResNet50---------------------------------------------------------------
#
model=torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# defining the loss function
criterion = nn.MSELoss()

# checking if GPU is available
# if torch.cuda.is_available():
#     model.cuda()
    # criterion = criterion.cuda().float()

print(model)

# =====================================================
print("Starting training loop...")
BATCH_SIZE = 256
training_loss = []
validation_loss = []
true_labels=[]
pred_labels =[]
training_rmse = []
testing_rmse =[]
for epoch in range(N_EPOCHS):
    loss_train = 0
    model.train().to(device)

    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds].to(device))
        # _, preds = torch.max(logits, 1)
        loss = criterion(torch.sigmoid(logits), y_train[inds].float())
        pred_labels.append(torch.sigmoid(logits).detach().cpu())
        true_labels.append(y_train[inds].detach().cpu())
        # print("preds",logits)
        # print("preds shape",logits.shape)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    training_loss.append(loss_train)
    # print("pred labels",pred_labels)
    train_rmse = get_rmse(torch.cat(true_labels).numpy(),torch.cat(pred_labels).numpy())
    training_rmse.append(train_rmse)
    print("Epoch {} --> Train Loss {:.5f}".format(epoch, loss_train))

    print("Training RMSE",train_rmse)
    # print(training_loss)
    torch.cuda.empty_cache()


    with torch.no_grad():
        model.eval()
        test_true_labels = []
        test_pred_labels = []
        y_test_pred = model(x_test.to(device))
        loss = criterion(torch.sigmoid(y_test_pred), y_test.float())

        test_true_labels.append(y_test.detach().cpu())
        test_pred_labels.append(torch.sigmoid(y_test_pred).detach().cpu())

        loss_test = loss.item()
    validation_loss.append(loss_test)
    test_rmse = get_rmse(torch.cat(test_true_labels).numpy(), torch.cat(test_pred_labels).numpy())
    testing_rmse.append(test_rmse)
    print("Test rmse",test_rmse)

    torch.save(model.state_dict(), "model_resnet.pt")
    print("Epoch {} --> Test Loss {:.5f} "+str(loss_test))
    # print(validation_loss)
    torch.cuda.empty_cache()


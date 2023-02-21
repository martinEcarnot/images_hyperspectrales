def main_loop(annot_dir, cnn, model_fn, labels_type, weights_loss, learning_rate, epochs=20, batch_size=12, other_class = False):
    """
    Main to train a model given a certain number of epoch, a loss and an optimizer

    :param cnn: class of the CNN to be used.
    :param annot_dir: path to the directory of the annotations files
    :labels_type: name of the column containing the labels, here "Face" or "Species"
    :param weights_loss: the weights to consider for each class
    :param learning_rate: Value for the exploration
    :param epochs: number of epochs used for training
    :param batch_size: number of image to process before updating parameters
    """
    bands = []
    with open(annot_dir + 'bands.txt', "r") as f:
        first_line = f.readline()
        second_line = f.readline().split(': ')[1]
        if second_line[:3] == 'RGB' :
            bands = [22, 53, 89]
        elif second_line[:3] == 'All':
            bands = [i for i in range(216)]
    dim_in=len(bands)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnn(dim_in).to(device)

    weight = torch.tensor(weights_loss).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    df_train = pd.read_csv(annot_dir + 'train_set.csv')
    if not other_class :
        df_train = df_train.loc[df_train['Face']!=2]
        df_train.index = [i for i in range(len(df_train))]
    print(df_train.head(10))

    train_set = CustomDataset(df_train, annot_dir, labels_type)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_valid = pd.read_csv(annot_dir + 'validation_set.csv')
    if not other_class :
        df_valid = df_valid.loc[df_valid['Face']!=2]
        df_valid.index = [i for i in range(len(df_valid))]

    print(df_valid.tail())
    val_set = CustomDataset(df_valid, annot_dir, labels_type)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    df_test = pd.read_csv(annot_dir + 'test_set.csv')
    
    if not other_class :
        df_test = df_test.loc[df_test['Face']!=2]
        df_test.index = [i for i in range(len(df_test))]
    
    test_set = CustomDataset(df_test, annot_dir, labels_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Nubmer of trainables parameters :" +str(pytorch_total_params))
    print('\nTraining model')
    list_accu_train = []
    list_accu_valid = []
    list_loss_train = []
    list_loss_valid = []
    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        model, correct, correct_valid, train_loss, valid_loss = train_model(train_loader, val_loader, device,
                                                                            model=model, loss_fn=loss_fn, optimizer=optimizer)
        list_accu_train.append(correct*100)
        list_accu_valid.append(correct_valid*100)
        list_loss_train.append(train_loss.item())  # Apparently tensor
        list_loss_valid.append(valid_loss)
        
    model_dir = os.path.join("models", model_fn)
    if not(os.path.exists(model_dir)):
        os.mkdir(model_dir)
    model_path = os.path.join("models",model_fn,model_fn + ".pth")
    print("\nSaving model at ", model_path)
    save_model(model, model_path)
    
    recap_path = os.path.join("models",model_fn,model_fn+"_summary.txt")
    with open(recap_path,'w') as recap_file:
        recap_file.write(summary_training(
            model, annot_dir, labels_type, weights_loss, learning_rate, epochs, batch_size, other_class, bands = bands))
    fig_fn = model_fn+"_training_evolution"
    display_save_figure(model_dir, fig_fn, list_accu_train, list_accu_valid, list_loss_train, list_loss_valid)

    print("\nTesting model")
    test_accu, test_loss = test_model(test_loader, device, model=model, loss_fn=loss_fn, test_dir = annot_dir, model_fn = model_fn)

    # Saving values
    print("\nSaving values of train, validation and test loops")
    save_array = np.asarray([list_accu_train, list_accu_valid, list_loss_train, list_loss_valid])
    np.savetxt(os.path.join(model_dir, model_fn +"_values_train_valid.csv"), save_array,
               delimiter=",", fmt='%.5e')  # Train
    np.savetxt(os.path.join(model_dir, model_fn +"_values_test.csv"), np.asarray([[test_accu], [test_loss]]),
               delimiter=",", fmt='%.5e')  # Test

    print("\nDone!")
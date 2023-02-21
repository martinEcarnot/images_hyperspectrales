from classification_face import *
from cnns import *

annot_dir = "img/cropped/"
i=13
j=3
weights_loss=[2.,2.]
learning_rates = [1e-4,5e-4]
model_fn_1="CNN_1_"
model_fn_1="CNN_2_"
n_epochs = 80
for lr in learning_rates:
    main_loop(
        annot_dir = annot_dir,
        cnn = CNN_1,
        model_fn = model_fn_1+str(i),
        labels_type = 'Face', 
        weights_loss = weights_loss,
        learning_rate = 1e-4,
        epochs=n_epochs,
        batch_size=32,
        other_class = False)
    i+=1
    main_loop(
        annot_dir = annot_dir,
        cnn = CNN_2,
        model_fn = model_fn_2+str(j),
        labels_type = 'Face', 
        weights_loss = weights_loss,
        learning_rate = 1e-4,
        epochs=n_epochs,
        batch_size=32,
        other_class = False)
    j+=1
"""
i=10
j=0
weights_loss=[2.,2.]
learning_rates = [1e-5,5e-4,1e-4,5e-4]
n_epochs = 80
for lr in learning_rates:
    main_loop(
        annot_dir = annot_dir,
        cnn = CNN_1,
        model_fn = model_fn_1+str(i),
        labels_type = 'Face', 
        weights_loss = weights_loss,
        learning_rate = lr,
        epochs=n_epochs,
        batch_size=32,
        other_class = False)
    i+=1
    main_loop(
        annot_dir = annot_dir,
        cnn = CNN_2,
        model_fn = model_fn_2+str(j),
        labels_type = 'Face', 
        weights_loss = weights_loss,
        learning_rate = lr,
        epochs=n_epochs,
        batch_size=32,
        other_class = False)
    j+=1"""
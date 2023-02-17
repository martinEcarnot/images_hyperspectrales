from classification_face import *
from cnns import *
annot_folder = "img/cropped/RGB"
learning_rates = [1e-5,5e-4,1e-4,5e-3,1e-3]
list_n_epochs = [40,50,60,70,80]
weight_loss = [2., 2.]
i = 0
model_fn = "CNN_1_"
for n_epochs in list_n_epochs:
    for lr in learning_rates:
        main_loop(annotations_folder,model_fn+str(i),'Face',weight_loss, learning_rate, epochs=epochs, batch_size=20, other_class = False)
        i+=1

from utils import *
from classification import *
from cross_validation import *
from cnns import *

"""
from classification_face import *
annot_dir = "img/cropped/RGB/"
learning_rate = 1e-4
epochs = 80
weights_loss = [2., 2.]
model_name = 'CNN_3_1'
main_loop(annot_dir = annot_dir, cnn = CNN_3, model_fn = model_name, labels_type = 'Face', 
              weights_loss = weights_loss, learning_rate = learning_rate, epochs=epochs, 
              batch_size=64, other_class = False #, chosen_var = [1, 8], chosen_face = 'Sillon'
          )
"""


"""
from classification_face import *
annot_dir = "img/cropped/"
learning_rate = 1e-4
epochs = 50
weights_loss = [2., 2.]
model_name = 'CNN_3_1_216'
main_loop(annot_dir = annot_dir, cnn = CNN_3, model_fn = model_name, labels_type = 'Face', 
              weights_loss = weights_loss, learning_rate = learning_rate, epochs=epochs, 
              batch_size=48, other_class = False #, chosen_var = [1, 8], chosen_face = 'Sillon'
          )
"""



cnn = CNN_2
model_fn = "CNN_2_cross_validation"
learning_rate = 1e-4
epochs = 80
labels_type = "Face"
weights_loss = [2., 2.]
batch_size = 64
other_class = False
K=5



cross_validation(
    annot_dir=annot_dir,
    cnn=cnn,
    model_fn=model_fn,
    labels_type=labels_type,
    weights_loss=weights_loss,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    other_class = other_class,
    K=K)



annot_dir = "img/cropped/RGB/"
cnn = CNN_3
model_fn = "CNN_3_cross_validation"
learning_rate = 1e-4
epochs = 80
labels_type = "Face"
weights_loss = [2., 2.]
batch_size = 64
other_class = False
K=5



cross_validation(
    annot_dir=annot_dir,
    cnn=cnn,
    model_fn=model_fn,
    labels_type=labels_type,
    weights_loss=weights_loss,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    other_class = other_class,
    K=K)



annot_dir = "img/cropped/"
weights_loss=[2.,2.]
model_fn="CNN_2_3"
n_epochs = 80

main_loop(
    annot_dir = annot_dir,
    cnn = CNN_2,
    model_fn = model_fn,
    labels_type = 'Face', 
    weights_loss = weights_loss,
    learning_rate = 1e-4,
    epochs=n_epochs,
    batch_size=48,
    other_class = False)


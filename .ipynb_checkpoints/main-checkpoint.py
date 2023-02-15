annot_folder = "img/cropped/"
#read_all_annot_csv(annot_folder)

from classification_face import *
annotations_folder = "img/cropped/"
learning_rate = 5e-4
epochs = 25
weight_loss = [2., 2.,2.]
main_loop(annotations_folder, 'Face',weight_loss, learning_rate, epochs=epochs, batch_size=8, other_class = False, bands = [22, 53, 89]
          )
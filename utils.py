import ast
import os
import random as rd
from os import listdir, walk
from os.path import join
from random import shuffle

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spectral as sp
from PIL import Image

from display_image import band_brightness


def clean_annot_csv(annot_dir, annot_fn):
    """
    Completes an annotations file which has been produced by crop_image.
    :param annot_dir: path to the annotations directory
    :param annot_fn: name of the annotations file
    """
    df = pd.read_csv(join(annot_dir, annot_fn + '.csv'))
    df.rename(columns = {'Label':'Face'}, inplace = True)
    df['Name_hdr'] = annot_fn[annot_fn.find("annotations_") + len("annotations_"):] + '_grain' + df.index.astype(str)+'.hdr'
    df['Species'] = df['Name_hdr'].str[3].astype(int)
    df.to_csv(join(annot_dir, annot_fn + '.csv'),index=False)

    
def read_all_annot_csv(annot_dir, out_fn='full_set',clean=True):
    """
    Concatenates the annotations files of all images and write them in a global annotations csv file.
    :param annot_dir: path to the annotations directory
    :param out_fn: name of the global annotations file (output)
    :param clean: if True, cleans each image's annotations file with clean_annot_csv()
    """
    df_tot = pd.DataFrame()
    for file in listdir(annot_dir):
        if file[:11]=='annotations' and file.split('.')[1]=='csv':
            path = join(annot_dir, file)
            if(clean):
                clean_annot_csv(annot_dir, file.split('.')[0])
            df = pd.read_csv(path)
            df_tot = pd.concat([df_tot,df])
    df_tot.to_csv(annot_dir + out_fn + '.csv', index=False)


def shuffle_full_set(annot_dir, annot_fn, out_dir):
    """
    Automatically executed during cross vlidation ; simply shuffles the full set
    without further dividing it into multiple datasets
    """
    df_full = pd.read_csv(annot_dir + annot_fn + '.csv')
    N = len(df_full)
    shuffled_indexes = [i for i in range(N)]
    shuffle(shuffled_indexes)
    df_full = df_full.iloc[shuffled_indexes]
    df_full.to_csv(join(out_dir, annot_fn + '.csv'), index=False)



def shuffle_train_val_test(annot_dir, prop=[0.7, 0.15, 0.15]):
    """
    Classic shuffle : 70% of data go into train set, 15% into validation_set, 15% into test_set
    """
    df = pd.read_csv('img/cropped/full_set.csv')
    N = len(df)
    shuffled_indexes = [i for i in range(N)]
    shuffle(shuffled_indexes)
    train_idx = shuffled_indexes[:int(prop[0]*N)]
    val_idx = shuffled_indexes[int(prop[0]*N):int((prop[1]+prop[0])*N)]
    test_idx = shuffled_indexes[int((prop[1]+prop[0])*N):]
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    
    df_train.to_csv(annot_dir + 'train_set.csv',index=False)
    df_val.to_csv(annot_dir + 'validation_set.csv',index=False)
    df_test.to_csv(annot_dir + 'test_set.csv',index=False)
    


 
def shuffle_leave_one_out(annot_dir, prop=[0.8, 0.2], spec_test = None): 
    """
    Select one specie for the test set ; the others are shuffled betxeen train nd val sets
    Possibility to choose the specie to leave out (parameter 'spec_test'). 
    By default, the specie to leave out is chosen randomly
    """  
    df = pd.read_csv('img/cropped/full_set.csv')
    if spec_test == None or spec_test not in [i for i in range(1, 9)]:
        spec_test = rd.randint(1, 8)
    df_test = df.loc[df['Species'] == spec_test]
    
    
    print("Espèce exclue de l'entraïnement : Espèce " + str(spec_test))
    shuffled_indexes = [i for i in df.loc[df['Species'] != spec_test].index]
    N = len(shuffled_indexes)
    shuffle(shuffled_indexes)
    train_idx = shuffled_indexes[:int(prop[0]*N)]
    val_idx = shuffled_indexes[int(prop[0]*N):]
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    df_train.to_csv(annot_dir + 'train_set.csv',index=False)
    df_val.to_csv(annot_dir + 'validation_set.csv',index=False)
    df_test.to_csv(annot_dir + 'test_set.csv',index=False)



def reconstitute_img(annot_dir_test_preds, annot_path_test_preds, img_folder):
    df = pd.read_csv(annot_dir_test_preds + annot_path_test_preds + '.csv')
    idx = rd.randint(0, len(df['Name_hdr']))
    img_name = df['Name_hdr'][idx].split("_grain")[0]
    df['Og_img'] = [df['Name_hdr'][i].split("_grain")[0] for i in range(len(df['Name_hdr']))]
    df = df.loc[df['Og_img'] == img_name]
    img = sp.open_image(img_folder + img_name + '.hdr')
    img_r = img[:, :, 22] / band_brightness(img, 22)
    img_g = img[:, :, 53] / band_brightness(img, 53)
    img_b = img[:, :, 89] / band_brightness(img, 89)
    img_fin = np.fliplr(cv2.rotate(np.dstack((img_b, img_g, img_r)), cv2.ROTATE_90_CLOCKWISE))
    print('Image : ' + img_name)
    fig, ax = plt.subplots()
    fig.set_figheight(50)
    fig.set_figwidth(50)
    ax.xaxis.tick_top()
    for i in df.index:
        bbox_list = ast.literal_eval(df['Bbox'][i])
        x1, y1, x2, y2 = bbox_list
        if df['Face_pred'][i] == 0 :
            color = 'blue'
        elif df['Face_pred'][i] == 1 :
            color = 'red'
        else :
            color = 'green'
        ax.add_patch(patches.Rectangle((y1, x1), y2 - y1, x2 - x1, fill=False, edgecolor=color, lw=2))
        plt.text(y2, x1, "{}".format(np.floor(max(ast.literal_eval(df['Probas'][i]))*100)/100), 
                 bbox={'facecolor' : color}, ha="left", va="bottom", fontsize = 16, color = 'w')
    plt.imshow(img_fin)
    plt.show()


def see_all_img(annot_dir, annot_path = 'full_set', img_folder = 'img/', labels_type = 'Face', preds = False, show_ind = False):
    df = pd.read_csv(annot_dir + annot_path + '.csv')
    df['Og_img'] = [df['Name_hdr'][i].split("_grain")[0] for i in range(len(df['Name_hdr']))]
    images_names = np.unique(np.array(df['Og_img']))
    for img_name in images_names :
        df_copy = df.loc[df['Og_img'] == img_name]
        img = sp.open_image(img_folder + img_name + '.hdr')
        img_r = img[:, :, 22] / band_brightness(img, 22)
        img_g = img[:, :, 53] / band_brightness(img, 53)
        img_b = img[:, :, 89] / band_brightness(img, 89)
        img_fin = np.fliplr(cv2.rotate(np.dstack((img_b, img_g, img_r)), cv2.ROTATE_90_CLOCKWISE))
        print('Image : ' + img_name)
        fig, ax = plt.subplots()
        fig.set_figheight(50)
        fig.set_figwidth(50)
        ax.xaxis.tick_top()
        if labels_type == 'Species':
                column = df['Species_pred']
        elif labels_type == 'Face':
                if preds :
                    column = df['Face_pred']
                else :
                    column = df['Face']
        lab = sorted(np.unique(column))
        for i in df_copy.index:
            bbox_list = ast.literal_eval(df['Bbox'][i])
            x1, y1, x2, y2 = bbox_list
            if column[i] == lab[0] :
                color = 'blue'
            elif column[i] == lab[1] :
                color = 'red'
            else :
                color = 'green'     
            ax.add_patch(patches.Rectangle((y1, x1), y2 - y1, x2 - x1, fill=False, edgecolor=color, lw=2))
            if preds :
                plt.text(y2, x1, "{}".format(np.floor(max(ast.literal_eval(df['Probas'][i]))*100)/100), 
                 bbox={'facecolor' : color}, ha="left", va="bottom", fontsize = 16, color = 'w')
            elif show_ind :
                plt.text(y2, x1, "{}".format(df.index[i]), 
                     bbox={'facecolor' : color}, ha="left", va="bottom", fontsize = 16, color = 'w')
        plt.imshow(img_fin)
        plt.show()
        fig.savefig(annot_dir + img_name + '_pred.png', dpi=200, format='png')
        

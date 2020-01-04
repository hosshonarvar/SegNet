import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def writeImage(image):
    """ store label data to colored image """
    Background = [0,0,0]
    Face = [200,200,200]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Background, Face])
    for l in range(0,2):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    plt.imshow(im)
    
def display_color_legend():
    
    Background          = np.array([0,0,0])/256
    Face     = np.array([200,200,200])/256

    patches = [mpatches.Patch(color=Background, label='Background'), mpatches.Patch(color=Face, label='Face')]
    
    plt.figure(figsize=(0.2,0.2))
    plt.legend(handles=patches, ncol=2)
    plt.axis('off')
    plt.show()
    
def draw_plots_bayes(images, labels, predicted_labels, uncertainty):
    
    num_images = len(images)
    
    cols = ['Input', 'Ground truth', 'Output', 'Uncertainty']
    rows = ['Image {}'.format(row) for row in range(1,num_images+1)]
    #rows = ['Worst', 'Average', 'Best']

    fig, axes = plt.subplots(nrows=num_images, ncols=4, figsize=(20,num_images*4))
    

    for i in range(num_images):

        plt.subplot(num_images, 4, (4*i+1))
        plt.imshow(images[i])
        #plt.ylabel("Image %d" % (i+1), size='18')
        plt.ylabel(rows[i], size='22')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[0], size='22', va='bottom')
        
        #print(labels[i])
        plt.subplot(num_images, 4, (4*i+2))
        # HH: scale 255 to 1 (face class)
        #labels[i] = np.divide(labels[i], 255)
        writeImage(labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[1], size='22', va='bottom')

        plt.subplot(num_images, 4, (4*i+3))
        writeImage(predicted_labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[2], size='22', va='bottom')
        
        plt.subplot(num_images, 4, (4*i+4))
        plt.imshow(uncertainty[i], cmap = 'Greys')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[3], size='22', va='bottom')

    plt.show()
    
    
def draw_plots_bayes_external(images, predicted_labels, uncertainty):
    
    num_images = len(images)
    
    cols = ['Input', 'Output', 'Uncertainty']
    rows = ['Image {}'.format(row) for row in range(1,num_images+1)]
    #rows = ['Worst', 'Average', 'Best']

    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(16,num_images*4))
    

    for i in range(num_images):

        plt.subplot(num_images, 3, (3*i+1))
        plt.imshow(images[i])
        #plt.ylabel("Image %d" % (i+1), size='18')
        plt.ylabel(rows[i], size='18')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[0], size='18', va='bottom')

        plt.subplot(num_images, 3, (3*i+2))
        writeImage(predicted_labels[i])
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[1], size='18', va='bottom')
            
        plt.subplot(num_images, 3, (3*i+3))
        plt.imshow(uncertainty[i], cmap = 'Greys')
        plt.xticks([])
        plt.yticks([])

        if (i==0): 
            plt.title(cols[2], size='18', va='bottom')

    plt.show()
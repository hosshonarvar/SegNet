# Image-Segmentation
Image segmentation of celebrity faces

In this project, I have developed a deep learning framework for image segmentation of celebrity faces. Below, I summarize the components of the project and more details can be found in the notebook file.

1-Dataset

- The Large-scale CelebFaces Attributes (CelebA) dataset in http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html is used to train and test the model.

- A dataset with 1400 images is used where the training dataset has 1000 images, the development dataset has 200 images, and the test dataset has 200 images.

- The dataset created for this project is in "/Face/1000Tr200Va200Te/" folder, which is currently empty due to the storage limit. The tar file for this folder ("Face.tar") can be downloaded from https://drive.google.com/file/d/1xUrSGHzlloKb_tGjONK6fyCn_1snAWjk/view?usp=sharing.


2-Training

- To develop the model, I used the implementation of SegNet neural network in http://github.com/toimcio/SegNet-tensorflow -- the SegNet was proposed in http://arxiv.org/abs/1511.00561 for segmentation of objects in an image.

- As part of the SegNet algorithm, the pretrained VGG-16 neural network is needed as the encoder. The vgg16.npy file can be downloaded from https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM, which should be in the root folder.

- The trained model is in "/Model/" folder, which does not have all the files due to the storage limit . The tar file for this folder ("Model.tar") with all the files can be downloaded from https://drive.google.com/file/d/1POYr3ce2o34_d0e5PpYp8-FjgTNYf0yP/view?usp=sharing.

- To run the code, Python 3.6.5 and Tensorflow 1.13.1 are used on a CPU machine.


3-Results

- Performance results: accuracy = 0.990128, mean IU = 0.977189, class # 0 accuracy = 0.995174, class # 1 accuracy = 0.978935

- The model uncertainties are calculated based on Bayesian SegNet in http://arxiv.org/abs/1511.02680.

<img src="Results.png">

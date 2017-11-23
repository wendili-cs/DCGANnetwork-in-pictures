# DCGANnetwork-in-pictures
Another type of GAN, using Deep Convolutional Generative Adversarial Nerworks.

## Introduction

 - The structure is based on Deep Convolutional Generative Adversarial Nerworks, worked on TensorFlow.
 
 - The whole training takes much time and calculation amount, maybe you'd better use GPU version or you will have to wait for a long time training.
 
 - Another choice is that you can download the trained model by me. I will upload the model trained from Celeb's 10000 face pictures.
 
 - Here is the hyperparameter:
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/250.jpg?raw=true)
 
The results using fully connected network is:
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/254.png?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/255.png?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/255.png?raw=true)
 
while using this DCGAN is :

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/251.png?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/252.png?raw=true)
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/253.png?raw=true)
 
there is much difference, right?
 
*********************

## How to use

 - Put all datasets in to a folder and change the ``input_data`` content to your dataset folder, After it trained, you can turn off To_Train button and run it again it will create pictures generated by the generator.

 - I will uploaded trained model to the folder `` 训练完成的模型 ``, whose name represent the dataset's type I trained. 
 Also, I will put the hyperparameter txt in it so that you should keep yours same to that and you can use it.

# Image Style Transfer Using Convolutional Neural Networks

## Intro

The goal of this project was to implement the 2016 paper "Image Style Transfer Using Convolutional Neural Networks" by Gatys, Ecker, and Bethge. Image style transfer involves capturing the “style” of an image and recreating another image using that style. This implementation uses different layers of a pre-trained CNN, [VGG-19](https://www.mathworks.com/help/deeplearning/ref/vgg19.html), to extract the semantic and stylistic features of images and merge them.

## Implementation
[VGG-19](https://www.mathworks.com/help/deeplearning/ref/vgg19.html) is a convolutional neural network that is 19 layers deep and trained on over 100 million images. It can be used for image classification, but in this implementation none of the classification features were used. Gradient descent to minimize the loss function given in the paper was aplied to the target image. As the loss function gets closer to zero, the image takes on more and more of the stylistic elements provided. The layers of VGG-19 used for the style portion of the image are conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1, each with their own respective weights, while conv4_2 is used for the content image. The optimizer used for gradient descent was L-BFGS as described in the paper.

## Results

To view in more detail, click an image

### Tuebingen-Neckarfront + The Shipwreck of the Minotaur

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="Tuebingen-Neckarfront" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront.png?raw=true) Tuebingen-Neckarfront | [<img width="1000" alt="The Shipwreck of the Minotaur" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/minotaur.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/minotaur.png?raw=true) The Shipwreck of the Minotaur |

| 50 epochs |  150 epochs | 250 epochs | 350 epochs  |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="Tuebingen-Neckarfront with The Shipwreck of the Minotaur style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/minotaur-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/minotaur-50.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with The Shipwreck of the Minotaur style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/minotaur-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/minotaur-150.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with The Shipwreck of the Minotaur style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/minotaur-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/minotaur-250.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with The Shipwreck of the Minotaur style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/minotaur-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/minotaur-350.png?raw=true) |

***

### Tuebingen-Neckarfront + The Scream

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="Tuebingen-Neckarfront" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront.png?raw=true) Tuebingen-Neckarfront | [<img width="1000" alt="The Scream" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/scream.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/scream.png?raw=true) The Scream |

| 50 epochs | 150 epochs | 250 epochs | 350 epochs |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="Tuebingen-Neckarfront with The Scream style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/scream-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/scream-50.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with The Scream style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/scream-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/scream-150.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with The Scream style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/scream-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/scream-250.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with The Scream style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/scream-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/scream-350.png?raw=true) |

***

### Tuebingen-Neckarfront + Starry Night

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="Tuebingen-Neckarfront" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront.png?raw=true) Tuebingen-Neckarfront | [<img width="1000" alt="Starry Night" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/starry-night.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/starry-night.png?raw=true) Starry Night |

| 50 epochs  | 150 epochs | 250 epochs | 350 epochs |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="Tuebingen-Neckarfront with Starry Night style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-50.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with Starry Night style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-150.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with Starry Night style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-250.png?raw=true) | [<img width="1000" alt="Tuebingen-Neckarfront with Starry Night style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-350.png?raw=true) |

***

### Texas A&M + The Shipwreck of the Minotaur

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="Texas A&M" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu.png?raw=true) Texas A&M | [<img width="1000" alt="The Shipwreck of the Minotaur" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/minotaur.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/minotaur.png?raw=true) The Shipwreck of the Minotaur |

| 50 epochs | 150 epochs | 250 epochs | 350 epochs |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="Texas A&M with The Shipwreck of the Minotaur style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/minotaur-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/minotaur-50.png?raw=true) | [<img width="1000" alt="Texas A&M with The Shipwreck of the Minotaur style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/minotaur-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/minotaur-150.png?raw=true) | [<img width="1000" alt="Texas A&M with The Shipwreck of the Minotaur style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/minotaur-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/minotaur-250.png?raw=true) | [<img width="1000" alt="Texas A&M with The Shipwreck of the Minotaur style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/minotaur-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/minotaur-350.png?raw=true) |

***

### Texas A&M + The Scream

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="Texas A&M" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu.png?raw=true) Texas A&M | [<img width="1000" alt="The Scream" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/scream.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/scream.png?raw=true) The Scream |

| 50 epochs | 150 epochs | 250 epochs | 350 epochs |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="Texas A&M with The Scream style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/scream-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/scream-50.png?raw=true) | [<img width="1000" alt="Texas A&M with The Scream style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/scream-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/scream-150.png?raw=true) | [<img width="1000" alt="Texas A&M with The Scream style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/scream-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/scream-250.png?raw=true) | [<img width="1000" alt="Texas A&M with The Scream style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/scream-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/scream-350.png?raw=true) |

***

### Texas A&M + Starry Night

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="tamu" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu.png?raw=true) Texas A&M | [<img width="1000" alt="Starry Night" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/starry-night.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/starry-night.png?raw=true) Starry Night |

| 50 epochs | 150 epochs | 250 epochs | 350 epochs |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="Texas A&M with Starry Night style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/starry-night-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/starry-night-50.png?raw=true) | [<img width="1000" alt="Texas A&M with Starry Night style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/starry-night-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/starry-night-150.png?raw=true) | [<img width="1000" alt="Texas A&M with Starry Night style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/starry-night-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/starry-night-250.png?raw=true) | [<img width="1000" alt="Texas A&M with Starry Night style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/starry-night-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tamu/starry-night-350.png?raw=true) |

***

### Giraffe + The Shipwreck of the Minotaur

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="giraffe" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe.png?raw=true) Giraffe | [<img width="1000" alt="The Shipwreck of the Minotaur" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/minotaur.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/minotaur.png?raw=true) The Shipwreck of the Minotaur |

| 50 epochs |  150 epochs | 250 epochs  | 350 epochs |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="giraffe with The Shipwreck of the Minotaur style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/minotaur-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/minotaur-50.png?raw=true) | [<img width="1000" alt="giraffe with The Shipwreck of the Minotaur style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/minotaur-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/minotaur-150.png?raw=true) | [<img width="1000" alt="giraffe with The Shipwreck of the Minotaur style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/minotaur-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/minotaur-250.png?raw=true) | [<img width="1000" alt="giraffe with The Shipwreck of the Minotaur style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/minotaur-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/minotaur-350.png?raw=true) |

***

### Giraffe + The Scream

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="giraffe" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe.png?raw=true) Giraffe | [<img width="1000" alt="The Scream" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/scream.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/scream.png?raw=true) The Scream |


| 50 epochs | 150 epochs | 250 epochs | 350 epochs |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="giraffe with The Scream style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/scream-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/scream-50.png?raw=true) | [<img width="1000" alt="giraffe with The Scream style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/scream-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/scream-150.png?raw=true) | [<img width="1000" alt="giraffe with The Scream style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/scream-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/scream-250.png?raw=true) | [<img width="1000" alt="giraffe with The Scream style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/scream-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/scream-350.png?raw=true) |

***

### Giraffe + Starry Night

| Content Image | Style Image |
|:-:|:-:|
| [<img width="1000" alt="giraffe" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe.png?raw=true) Giraffe | [<img width="1000" alt="Starry Night" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/starry-night.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/starry-night.png?raw=true) Starry Night |

| 50 epochs |  150 epochs | 250 epochs  | 350 epochs |
|:-:|:-:|:-:|:-:|
| [<img width="1000" alt="giraffe with Starry Night style after 50 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/starry-night-50.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/starry-night-50.png?raw=true) | [<img width="1000" alt="giraffe with Starry Night style after 150 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/starry-night-150.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/starry-night-150.png?raw=true) | [<img width="1000" alt="giraffe with Starry Night style after 250 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/starry-night-250.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/starry-night-250.png?raw=true) | [<img width="1000" alt="giraffe with Starry Night style after 350 epochs" src="https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/giraffe/starry-night-350.png?raw=true">](https://github.com/BrianGlasheen/Image-style-transfer-CNN/blob/main/images/tuebingen-neckarfront/starry-night-350.png?raw=true) |

***

In these examples you can see that the program struggles because of the low levels of semantic content present in giraffe image. I believe this is because the content image boils down to subject (the giraffe), background, and foreground. With not much granularity in the semantic content the model applies the style in non ideal ways, I hypothesize.

## Conclusion

Some images retain their semantic information (the content of the image) better while others get blended together more easily. There is not a one size fits all set of parameters that produces a perfect image. The original paper was published in 2016, and it’s impressive how good the process already was, before AI’s rise in recent years. The training process is easy to modify, simply pass in a number of epochs along with a max iterations and learning rate for the optimizer. More epochs allows the program to perform gradient descent longer, images are saved every 50 iterations. A higher learning rate will make the neural network take larger steps on each iteration. The max iterations is the number of jumps that the L-BFGS will take when minimizing during each epoch.

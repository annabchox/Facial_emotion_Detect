# Facial Emotion Detector
----
## Problem Statement

Many individuals with visual impairment, facial recognition impairments, or autism face challenges in recognizing and understanding emotions, which can be crippling in social interactions and emotional communication. Many of these individuals predominantly rely on other cues, such as tone of voice and context. For autistic individuals, many studies show that one can learn how to interpret emotions through practice. [Source](https://luxai.com/blog/emotion-recognition-for-autism/) However, many other individuals with more severe visual impairments such as facial blindness may not be able to see facial patterns to interpret human emotions. Therefore, there is a need for an inclusive and accessible solution that empowers individuals with these conditions to accurately perceive and interpret emotions.

### Objective:

The objective of this project is to develop a facial emotion recognition model. This model could have the potential to assist individuals with visual impairment, facial recognition impairments, or autism in perceiving and understanding emotions.

### Success Criteria:

The success of this project will be evaluated based on the following criteria:

- Accuracy and Performance: The emotional recognition model should demonstrate high accuracy and robustness in classifying emotions across different input modalities, providing reliable results to users.


----
## Folder Directory
|Folder Name|File Name|File Description|
|---        |---      |---             |
|Code|| This folder contains all code for the project
|Code/Helper|| This folder contains all the helper scripts.
|Code/Helper|`helper.py`| This files contains all the helper functions and variables used in the `.ipynb` files
|Code/Preprocessing|| This folder contains all preprocessing and cleaning steps
|Code/Preprocessing|`1_FER2013.ipynb`| This folder contains all preprocessing and cleaning steps on the FER2013 dataset
|Code/Preprocessing|`2_AffectNet.ipynb`| This folder contains all preprocessing and cleaning steps on the AffectNet dataset
|Code/Preprocessing|`3_Final_Data.ipynb`| This folder contains all preprocessing and cleaning steps to obtain the Final dataset.
|Code/Training|| This folder contains all the main modeling work on training and validation.
|Code/Training|`1_Training_CNN.ipynb`| This folder contains all the main CNN model work.
|Code/Training|`2_EfficientNet.ipynb`| This folder contains all the main EfficientNet model work.
|Code/Testing|| This folders contains modeling work on testing
|Code/Testing|`Best_Model.ipynb`| This file contains the best trained model applied to testing data and evaluations
|Results|| This folder contains the summary results from the models.
|Results|`model_eval.csv`| This file contains the summary of main model results.
|Models|| This folder contains all the pre-trained models. (Not included because files are too large)
|Streamlit|| This folder contains the script to deploy the best model on Streamlit.
|Images|| This folder contains all the images saved and used in the presentation.
|Data|| This folder is empty because data is not uploaded; but is where relative filepaths from training notebooks should be

----
## Data Collection and Preprocessing

Data is taken from [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013), which is an open-source dataset used in the paper "Eavesdrop the Composition Proportion of Training Labels in Federated Learning" by Wang, Lixu  et al. Not much cleaning was needed besides splitting the data into train, val, and test datasets.

- More data had to be acquired due to major class imbalance in the FER-2013 dataset.
- The dataset consisted of 7 classes: angry, disgust, fear, happy, neutral, sad, surprise


[Class Dist FER2013](https://github.com/annabchox/facial_emotion_detector/assets/112204360/2f8dcfdb-5ceb-498c-b2d4-cfe5094e77ef)
 FER2013.jpg)


Additional data was acquired from [AffectNet](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data), which is a subset off data from an open-source dataset used in the paper "Landmark Guidance Independent Spatio-channel Attention and Complementary Context Information based Facial Expression Recognition" by Gera, Darshan et al. 

- The dataset consisted of 8 classes: anger, contempt, disgust, fear, happy, neutral, sad, surprise

Preprocessing for modeling included image formatting to (48, 48), scaling (done within the layers), ensuring all images were grayscale, and dropping the `contempt` class from AffectNet since this was not a class present in the FER-2013 dataset.

Once the AffectNet dataset was preprocessed to match the FER-2013 images, they were combined and further data augmentation was applied to the minority class, `disgust` to account for the continued class imbalance. After data augmentation was applied to the `disgust` class, the final dataset consisted of 7 classes with a total of 57,677 images.

| Class         | Percentage |
| ------------- |:----------------:|
| Angry         | 14.28%          |
| disgust       | 14.29%           |
| fear          | 14.29%      |
| happy        | 14.29%           |
| neutral     | 14.29%          |
| sad     | 14.29%           |
| surprise     | 14.28%          |

![Class Dist Final](https://github.com/annabchox/facial_emotion_detector/assets/112204360/b10c272d-bcab-4721-a8df-bb24f8b41e74)


----
## Modeling

In total, 8 different iterations of CNNs were used across 2 architectures (custom Sequential and EfficientNetV2B0).

----
## Conclusion and Recommendations 

Based on our findings, we recommend using a custom CNN ensemble model. This model provided the highest scores of validation accuracy of 64.43% and accuracy score on the test dataset of 63.15%.
![Screenshot 2023-06-12 at 10 01 11 AM](https://github.com/annabchox/facial_emotion_detector/assets/112204360/f5d8c2c6-0023-4872-8caf-8d1a72b37e7c)

----
## Uncertainties

While the ensemble model performed the best in terms of validation accuracy, it had an large validation loss. There is definitely more room to grow in creating a more robust and better performing model. Since the original FER-2013 dataset has a 48 x 48 resolution, it may be a good idea moving forward in to train a model on higher resolution images that are in color. It may be that the model had a difficult time extracting patterns and features for different emotion classes. Once a more robust model is created, applying the model on Streamlit with OpenCV and Haar Cascade Classifier could be extremely useful in aiding individuals classify facial emotions in real-time. 
Another important note is that while additional data and data augmentation to the `disgust` class was applied to the training data, further alterations or additions were not applied to the testing data. This means that the testing data had very imbalanced classes. This can help evaluate the performance of a model more effectively by giving a better comparison to how it perform on the training data.

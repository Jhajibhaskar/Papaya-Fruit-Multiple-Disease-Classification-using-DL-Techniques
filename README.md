# Papaya-Fruit-Multiple-Disease-Classification-using-Deep Learning-Techniques
Papaya diseases can significantly impact crop yield and quality. Automated disease detection through image classification can aid farmers in early identification and management of these diseases.<br>
This project focuses on classifying multiple diseases in papaya fruits using deep learning techniques. The goal is to distinguish between healthy and various diseased papayas to aid in early detection and management of papaya diseases. The dataset consists of images categorized into six classes: healthy, anthracnose, phytophthora blight, brown spot, black spot, and others.

## Dataset
The dataset includes images of papaya fruits, each classified into one of six categories:<br>

0. Healthy<br>
1. Anthracnose<br>
2. Phytophthora blight<br>
3. Brown spot<br>
4. Black spot<br>
5. Others<br>

Each category contains 650 images for training and 100 images for testing.

## Implementation Overview
### 1. Data Curation and Pre-Processing:
1. Data Collection: Images containing multiple fruits were removed to ensure each image represents a single papaya.
2. Data Distribution: The images were equally distributed across six classes to ensure balanced training data.
3. Pre-Processing: Standard image pre-processing techniques were applied, including resizing, normalization, and augmentation.
### 2. Model Evaluation:
#### Initial Evaluation
We initially developed a custom Convolutional Neural Network (CNN) architecture, which yielded:<br>
➼Training Accuracy: 76.41%<br>
➼Testing Accuracy: 60.33%

➢Recognizing the potential of pre-trained models, we experimented with eight established architectures for our classification task:
1. AlexNet<br>
2. DenseNet<br>
3. EfficientNet<br>
4. InceptionNet<br>
5. MobileNet<br>
6. NASNet<br>
7. ResNet<br>
8. VGGNet<br>

We evaluated eight pre-trained architectures including AlexNet, DenseNet, EfficientNet, InceptionNet, MobileNet, NASNet, ResNet, and VGGNet.
Based on initial evaluations, MobileNet emerged as the most promising model due to its superior performance.
#### Pre-Trained Model Performance
![image](https://github.com/Jhajibhaskar/Papaya-Fruit-Multiple-Disease-Classification-using-DL-Techniques/assets/84240276/c3a2b529-4691-4bb7-92c9-fc8a20c544f0)
#### Model Fine-Tuning

We fine-tuned the later layers of MobileNet and our custom CNN model to enhance their performance.

##### Fine-Tuning Results
![image](https://github.com/Jhajibhaskar/Papaya-Fruit-Multiple-Disease-Classification-using-DL-Techniques/assets/84240276/f1ddb499-cce5-48ac-ba5f-d4845ad6bbd8)

### 3. Model Selection:
➢Among the evaluated pre-trained models, MobileNet initially showed promising results; however, after meticulous fine-tuning, our custom CNN achieved the highest testing accuracy of 84.33%, making it the final choice for this project due to its superior performance.
### 4. Deployment:
➢Hosted our fine-tuned CNN model on the web using Streamlit for easy access and to interact the users with the model.<br>
➢Explore the deployed CNN model interface here: https://jhajibhaskar4.streamlit.app/
## Results
Our experiments demonstrated the effectiveness of leveraging pre-trained models, with the fine-tuned custom CNN achieving a notable testing accuracy of 84.33% and training accuracy of 89.41%, significantly surpassing initial performance.
#### Accuracy Graph - fine-tuned CNN
![image](https://github.com/Jhajibhaskar/Papaya-Fruit-Multiple-Disease-Classification-using-DL-Techniques/assets/84240276/1a4027c6-2168-4ae7-9f49-1b65d491ea0b)









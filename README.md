# NLP-TextArticle
 Training 2,200 text data to categorize the text article into 5 categories.
 
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

# Project Title
The prediction of categorizing the article into different categories of genre. 

## Description
In this project, Neural Network approach was used to conduct the NLP project. The dataset consisting of texts was trained to categories the unseen articles into 5 categories namely Sport, Tech, Business, Entertainment and Politics. The exploratory data analysis was conducted to remove the duplicates, unknown characters, numbers and standardising the text articles into lower-case in the datasets. The OneHotEncoder was performed on the target vaiable as well as tokenization and padding step was performed on texts prior to the model creation and development steps. After the training has completed, the trained data was deployed to test the model.

## Results and Deployment
In the pre-processing steps, the data was transformed and fitted in a few steps such as Train-split test, One Hot encoding, tokenization and padding methods. In the model development procedure, the model was created with additions of embedding layer, bidirectional layer as well as dense layer to create the model as shown in Model's architecture below.
![model's architecture](https://user-images.githubusercontent.com/107612253/175307159-edc11ff4-f67d-438b-9c8d-e21cb60f2e48.png)

The created model was trained with a batch size of 64 and epochs value of 100 to achieve an accuracy of 92% in f1 score as shown in the model's evaluation result.

![Model Evaluation](https://user-images.githubusercontent.com/107612253/175307541-1832a535-1370-4b96-957c-56dd93c621e0.jpg)

The model achieved 92% in accuracy in categorizing the unseen articles into  5 categories namely Sport, Tech, Business, Entertainment and Politics.  Despite the additions of nodes, LSTM and Bidirectional layers in the model’s creation, the model's performance improved significantly but the learning curve was overfit after multiple trainings with adjustments on the model creation as visualized in the Tensorboard and Spyder. The learning curve was overfitted as indicated by a large distanct between the training and validation losses without flattening as the training increases. As for the accuracy,training accuracy is slightly higher than validation accuracy, which is also typical to an overfit model.To overcome the overfitting learning model, adding additional training data may help. 

![Tensorboard's accuracy plot](https://user-images.githubusercontent.com/107612253/175308331-5329a285-8f30-41ca-89b1-56611e76f057.jpg)
![Tensorboard's loss plot](https://user-images.githubusercontent.com/107612253/175308355-ad486a97-4433-4b31-b2a8-6c7355757b80.jpg)

![Spyder's Acc plot](https://user-images.githubusercontent.com/107612253/175308397-a2cce491-ed88-4f7f-bcdf-e6d0570000d9.png)
![Spyder's Loss plot](https://user-images.githubusercontent.com/107612253/175308429-b189e3cb-57f8-46c2-94f9-f4f37f24fd75.png)



Despite slightly overfitted, the model can be considered a good predictive model as indicated by the f1 score with a value of 0.92 which is also proven in the deployment of the model.
![Deployment of the model](https://user-images.githubusercontent.com/107612253/175308717-fed6fe54-03f0-48ec-b394-25fba3a9f391.png)


## Acknowledgement
I would like to sincerely thank  Miss Susan Li forcontributing the dataset which can be downloaded from [github](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv). I sincerely enjoyed working on this project.:smiling_face_with_three_hearts:	:smiling_face_with_three_hearts:	:smiling_face_with_three_hearts:	

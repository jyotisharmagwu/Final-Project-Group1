#### This Folder contains the code for the project.
#### All the coding file are written in .py format using PyCharm

## Below are the steps to follow to run code

### 1) Dataset Download:
To download the dataset to your cloud we used Kaggle API commands.
Please follow the steps of instructions mentioned in pdf file named -> Dataset Upload and Download steps.pdf

#### Please make sure to unzip all the folders named below before running the code on your terminal:
 1) images_training_rev1
 2) training_solutions_rev1
 3) training_solutions_rev1
 
 ### 2) Path change for running model trainfiles and predict files:
 we have build three different training models using CNN named as below:
  - model__train_cnn.py
  - model__train_dataaugumentation_cnn.py
  - model__train_resnet.py
  - predict_testing.py
  
  For running above highlighted training and testing files, please change the path to the path of your directories where the upzipped     data files/folders of the dataset saved in your terminal:
  
  #### In Data Import section For all train files (model__train_cnn.py, model__train_dataaugumentation_cnn.py, model__train_resnet.py)
  ==>  df = pd.read_csv("  Please provide the path of csv file named (training_solutions_rev1.csv) from your terminal") 
  
  ==>  image = cv2.imread(" Please provide the path of testing folder named (images_training_rev1) " + i + '.jpg') 
  
   
  ### For testing predict file (predict_testing.py) make changes for paths
  ##### In  Data Loading section make changes for below :
  ==>  path_file_2 = ' Please provide the path of testing folder named (testing_images) here '
  
  ==> img = cv2.imread("Please provide the path of testing folder named (testing_images)" + i)
  
  ##### In "Submission CS FOR LEADERSHIP" section make changes for below :
  ==>  df = pd.read_csv('Please provide the path of csv file named (training_solutions_rev1.csv) from your terminal')
  
  ==>  val_files = os.listdir('Please provide the path of testing folder named (testing_images)')
  
  
  # Evaluation:
  Our analysis says that after applying three kind of models we found our custom model with data augumentation makes more sense. RMSE for training and validations set are giviung us a smooth curve.
  
  # ResNet saved model and csv submission file for kaggle competion output is on the below shareable link.(Open access through GWU Email    ID ONLY)
  https://drive.google.com/open?id=1Qn4Ul0LOoKytj1oC7PeAtNtTKFq7YKg1
  

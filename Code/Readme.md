#### This Folder contains the code for the project.
#### All the coding file are written in .py format using PyCharm

## Below are the steps to follow to run code

### 1) Dataset Download:
To download the dataset to your cloud we used Kaggle API commands.
Please follow the steps of instructions mentioned in pdf file named -> Dataset Upload and Download steps.pdf

#### Please make sure to unzip all the folder names below before running the code:
 1) images_training_rev1
 2) training_solutions_rev1
 3) training_solutions_rev1
 
 #### 2) Path change for running model trainfiles and predict files:
 we have build three different training models using CNN named as below:
  - model__train_cnn.py
  - model__train_dataaugumentation_cnn.py
  - model__train_resnet.py
  - predict_testing.py
  
  For running above highlighted training and testing files, please change the path to the path of your directories where the upzipped     data files/folders of the dataset saved in your terminal:
  
  #### For all model_train_ files 
  ==>  df = pd.read_csv("  Please provide the path of csv file named (training_solutions_rev1.csv) from your terminal") 
  
  ==>  image = cv2.imread(" Please provide the path of testing folder named (images_training_rev1) " + i + '.jpg')
  
  #### For predict_testing file 
  ==>  path_file_2 = ' Please provide the path of testing folder named (testing_images) here '
  
  ==>  df = pd.read_csv('Please provide the path of csv file named (training_solutions_rev1.csv) from your terminal')
  
  ==>  val_files = os.listdir('Please provide the path of testing folder named (testing_images)')
  
  
  
  

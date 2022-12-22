# ML Project 2

## Exploring deep learning models to detect sleep apnea episodes using the Physionet Apnea-ECG Database.
This project aims to find the best classification model to detect sleep apnea from ECG signal recordings of different subjects. The best results were obtained with 1D convolutional neural network.

### Files Description:
1. ```base_model\base_model.ipynb``` In this notebook we are training and evaluating the base model of our project which is a logistic regression on two features related to heart beats, bpm (beats per minute) and sdnn (standard deviation of the NN (R-R) intervals)
2. ```base_model\base_model_test_set.csv``` Test dataset for the baseline model, it consists of labels (apnea or not) and two features(bpm and sdnn).
3. ```base_model\base_model_train_set.csv``` Train dataset for the baseline model, it consists of labels (apnea or not) and two features(bpm and sdnn).
4. ```base_model\create_dataset_base_model.ipynb``` In this notebook we create the datasets used in our base model
5. ```base_model\utils.py``` Diverse utility functions
6. ```ECG_Data\apnea-ecg-database-1.0.0``` This is the ECG recordings data set, for more information on it check https://physionet.org/content/apnea-ecg/1.0.0/
7. ```ECG_Data\outputs``` These are the apnea annotations at each moment in time for each recording in a separate file txt.
8. ```google_colab_notebooks\1D_CNN.ipynb``` This notebook implements, trains and evaluates a 1D CNN model to detect sleep apnea from 1 minute signal segments.
9. ```google_colab_notebooks\1D_CNN_LSTM.ipynb``` This notebook implements, trains and evaluates a 1D CNN model with LSTM to detect apnea from 1 minute signal segments, thus accounting for the time-dependency between segments.

10. ```google_colab_notebooks\rp_CNN_google_net.ipynb``` This notebook uses google-net for transfer learning to make predictions of apnea on the recurrence plots images generated from 1 minute signal segments
11. ```google_colab_notebooks\spect_2D_CNN.ipynb``` This notebook implements, trains and evaluates a 2D CNN to detect apnea from spectrogram images generated from 1 minute signal segments
12. ```google_colab_notebooks\spect_CNN_google_net.ipynb``` This notebook uses google-net for transfer learning to make predictions of apnea on the spectrograms images generated from 1 minute signal segments

13. ```images_creation\create_recurrence_plots.ipynb``` In this notebook we create the recurrence plot images, and mapping to labels in a csv, that are used for the google_net model
14. ```images_creation\create_spectrograms.ipynb``` In this notebook we create the spectrogram images, and mapping to labels in a csv, that are used for the google_net model and 2D CNN model

15. ```images_creation\utils.py``` Diverse utility functions



### How: 

#### To perform the pipeline of the base model :
1. Run the ```base_model\base_model.ipynb```, it will train and make predictions on the already generated csv's: ```base_model_test_set.csv``` and ```base_model_train_set.csv```.           
2. In case you would like to generate the datasets again, run the ```create_dataset_base_model.ipynb```.


#### To generate the spectrograms and recurrence plots:

1. run the ```images_creation\create_recurrence_plots.ipynb``` and ```images_creation\create_spectrograms.ipynb```, each of these scripts will create a folder for each spectrogram/recurrence_plots image and an csv that maps each image name to a label. Note it can take significant time to generate.

#### To run the deep learning notebooks: 

1. Use this link : https://drive.google.com/drive/folders/1Re6xYXUmrhiL9BZKLP32yYfAOGp7qlX5?usp=sharing
This link provides all the data, it has the raw signals downloaded from Physionet as well as the spectrograms and recurrence plots in separate zip folders. The shared google drive folder is named "ML". From this shared folder, create a short cut to the main folder ("My Drive") in your google drive. You can do this by dragging/dropping the shared folder to "My Drive" (main google drive folder)

2. Run any of 5 notebooks present in ```google_colab_notebooks``` with google colab with the same account where the shared folder is present. It will ask you to connect to google drive to mount the data, proceed so. If step 1 has been done correctly and you have available gpu on google colab, the models should train and make the same predictions. If you have some unexpected problems with the drive of the path, it is always possible to change the directory name ```/My Drive/``` to make it fit your google drive folder. 

This folder will be used to do localized training of the model on 70% training dataset and test on the 30% testing dataset.
This will be used as the base case to show conventional model training.

Folder structure breakdown:
base_case_folder : contains the base case model for local training (conventional model training)
test_covid_folder : contains test images
train_covid_folder : contains train images
metadata.csv : csv reference to image metadatas (covid or no covid and filepath)
- 30 represents the 30% split data
- 70 represents the 70% split data

How to run the program:
1) open cmd : python main_process.py
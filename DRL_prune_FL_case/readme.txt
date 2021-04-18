Federated learning process
Folder system description:
Federated_learning_case : This folder will house the server script for the federated learning process
DRL_FL_client : This folder will house the client script for the federated learning process

Procedure to reproduce result
Server side:
1) Enter to server_main.py to create and train on base global model
2) Add base global model path to websocket_server.py script
3) Change host IP address and port number on the orchestrator.py script (Federated_learning_case\orchestrator)
4) Run the orchestrator script
- this will initiate the server script in a run till cancel loop

Client side:
1) Change the URI address to the server IP address in the client_code.py (DRL_FL_client)
2) Run the client_code.py

Additional notes:
1) Image dataset is not provided, preprocessing such as resizing is done in the pytorch_mobilenetv2_model.py script
- Add the training images to the train_covid_folder
- Add the testing images to the test_covid_folder
2) If the Client cannot connect to the server (Noticeable especially when the client script does not produce response) try the following:
- Turn off the firewall on the server host machine
- Make sure the port of the server host machine is actively listening (do a ping)
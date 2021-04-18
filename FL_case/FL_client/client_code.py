'''
Some side notes:
1) comment is alt+3 uncomment is alt+4
2) backslash, change the keyboard type to US under the pi setting.
'''
import torch
import os

import cifarnet as cifar
import MobileNetV1 as mobilev1
import websockets
import asyncio
import time
import sys

model_type = 'cfn'

if model_type == 'cfn':
    os.makedirs('./cfn',exist_ok=True)
    model_save_path = './cfn/cifarnet_recv.pth'
    train_save_path = './cfn/mv1_trained.pth'
elif model_type == 'mv1':
    os.makedirs('./mv1', exist_ok=True)
    model_save_path = './mv1/mobilenetv1_recv.pth'
    train_save_path = './mv1/mv1_trained.pth'

def main():
    global model_save_path
    global model_type

    if model_type == 'cfn':
        print('Initiating FL process')
        train_dataloader = cifar.load_dataset()
        model = cifar.get_net()
        print('Localized data initialized.....')
        recv_start_time = time.time()
        print('Attempting to connect and receive data from server.....')
        asyncio.get_event_loop().run_until_complete(connect_recv())
        recv_end_time = time.time()
        recv_time = recv_end_time - recv_start_time
        print(
            'Data received from server. Time taken : {:.10} seconds OR {:.5} minutes'.format(recv_time, recv_time / 60))

        print('Loading the model....')
        model = cifar.load_model(model, model_save_path)
        train_start_time = time.time()
        print('Training the model locally......')
        cifar.train_model(model, train_dataloader,train_save_path)
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        print('Finished training the model, sending the model over to the server for aggregation...')

        send_start_time = time.time()
        asyncio.get_event_loop().run_until_complete(connect_send())
        send_end_time = time.time()
        print('Process officially finished')
        send_time = send_end_time - send_start_time
        print('Time taken to send model : {:.10} seconds or {:.5} minutes'.format(send_time, send_time / 60))
        os.makedirs('./cfn',exist_ok=True)
        with open('./cfn/Report.txt', 'a') as outs:
            outs.write('Time taken for receiving model from server : {:.10} Seconds OR {:.5} Minutes\n'.format(recv_time,
                                                                                                             recv_time / 60))
            outs.write(
                'Time taken for training model : {:.10} Seconds OR {:.5} Minutes\n'.format(train_time, train_time / 60))
            outs.write('Time taken for sending model to server : {:.10} Seconds OR {:.5} Minutes\n'.format(send_time,
                                                                                                         send_time / 60))

    elif model_type == 'mv1':
        print('Initiating FL process')
        train_dataloader = mobilev1.load_dataset()
        model = mobilev1.get_net()
        print('Localized data initialized.....')
        recv_start_time = time.time()
        print('Attempting to connect and receive data from server.....')
        asyncio.get_event_loop().run_until_complete(connect_recv())
        recv_end_time = time.time()
        recv_time = recv_end_time - recv_start_time
        print('Data received from server. Time taken : {:.10} seconds OR {:.5} minutes'.format(recv_time,recv_time/60))

        print('Loading the model....')
        model = mobilev1.load_model(model,model_save_path)
        train_start_time = time.time()
        print('Training the model locally......')
        mobilev1.train_model(model,train_dataloader,train_save_path)
        train_end_time = time.time()
        train_time = train_end_time - train_start_time
        print('Finished training the model, sending the model over to the server for aggregation...')

        send_start_time = time.time()
        asyncio.get_event_loop().run_until_complete(connect_send())
        send_end_time = time.time()
        print('Process officially finished')
        send_time = send_end_time - send_start_time
        print('Time taken to send model : {:.10} seconds or {:.5} minutes'.format(send_time,send_time/60))
        os.makedirs('./mv1', exist_ok=True)
        with open('./mv1/Report.txt','a') as outs:
            outs.write('Time taken for receiving model from server : {:.10} Seconds OR {:.5} Minutes\n'.format(recv_time,recv_time/60))
            outs.write('Time taken for training model : {:.10} Seconds OR {:.5} Minutes\n'.format(train_time,train_time/60))
            outs.write('Time taken for sending model to server : {:.10} Seconds OR {:.5} Minutes\n'.format(send_time,send_time/60))

    
    
async def connect_recv():
    global model_type
    global model_save_path
    uri = 'ws://192.168.1.149:8765'
    if os.path.isfile(model_save_path):
        os.remove(model_save_path)
    async with websockets.connect(uri) as websocket:
        if model_type == 'cfn':
            await websocket.send('cfn')
        elif model_type == 'mv1':
            await websocket.send('mv1')

        model_byte_dict = {}
        counter = 0
        while(1):
            response = await websocket.recv()
            if response == 'end':
                break
            else:
                print('This is the size of the {} byte : {}'.format(counter,sys.getsizeof(response)))
                model_byte_dict[counter] = response
                counter += 1
                
        with open(model_save_path,'wb') as outs:
            for chunk_id,m_bytes in model_byte_dict.items():
                outs.write(m_bytes)
        

async def connect_send():
    global model_type


    uri = 'ws://192.168.1.149:8765'
    async with websockets.connect(uri) as websocket:
        if model_type == 'cfn':
            await websocket.send('aggregate_cifarnet')
        elif model_type == 'mv1':
            await websocket.send('aggregate_mv1')
        response = await websocket.recv()
        if response == 'aggregate recv':
#             send the chunk over at a time
            chunk_dict = chunking()
            for part,bytes_section in chunk_dict.items():
                await websocket.send(bytes_section)
            await websocket.send('end')
            response = await websocket.recv()
            print(response)
            
def chunking():
    global train_save_path
    with open(train_save_path,'rb') as model_stream:
        model_bytes = model_stream.read()
    chunking_split = 2**19
    
    chunk_dict = dict()
    counter = 0
    for i in range(0,len(model_bytes),chunking_split):
        chunk_dict[counter] = model_bytes[i:i+chunking_split]
        counter += 1
    return chunk_dict
    
    

main()
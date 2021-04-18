

# WS server example

import asyncio
import websockets
import pickle
import torch
import sys
import os
import math
import sys
os.chdir('../')
sys.path.append(os.getcwd())
print(os.getcwd())
print(sys.path)

# from server.model_scripts.pytorch_lenet_model import get_net as  get_lenet
# from server.model_scripts.pytorch_mobilenetv2_model import get_net as  get_mv2net
# from server.model_scripts.pytorch_mobilenetv2_model import save_model
# from server.model_scripts.pytorch_inception_model import get_net as get_iv3net
from server.model_scripts.cifarnet import get_net as get_cifarnet
from server.model_scripts.MobileNetV1 import get_net as get_mv1net

class WebsocketServerWorker():
    def __init__(self,
                 host : str,
                 port : int,
                 loop= None
                 ):

        self.host = host
        self.port = port
        self.nextMsg = False
        self.msgInfo = None
        self.agg = False
        self.modelBytes = {}

        if loop is None:
            loop = asyncio.new_event_loop()
        self.loop = loop

        self.broadcast_queue = asyncio.Queue()

        self.connected_clients = set()
        self.lock = asyncio.Lock()


    #consumer handler
    async def _consumer_handler(self,websocket : websockets.WebSocketCommonProtocol):
        """This handler listens for messages from WebsocketClientWorker
               objects.
           Args:
               websocket: the connection object to receive messages from and
                   add them into the queue.
        """
        try:
            self.connected_clients.add(websocket)
            while True:
                msg = await websocket.recv()

                await self.broadcast_queue.put(msg)
        except websockets.exceptions.ConnectionClosed:
            self._consumer_handler(websocket)

    #producer handler
    async def _producer_handler(self, websocket : websockets.WebSocketCommonProtocol):
        """This handler listens to the queue and processes messages as they
                arrive.
           Args:
                websocket: the connection object we use to send responses
                           back to the client.
        """

        while True:
            #get the message from the queue
            message = await self.broadcast_queue.get()
            # print(message)

            if isinstance(message, str):
                self.nextMsg = False
                message = message.strip()

            if message == 'mv1':
                print('Fetching mv1net on request...')
                # critical section
                await self.lock.acquire()
                chunk_dict = self.process_chunking('mv1')
                self.lock.release()
                for chunk_idx, byte in chunk_dict.items():
                    await websocket.send(byte)
                await websocket.send('end')

            elif message == 'cfn':
                print('Fetching cifarnet on request...')
                # critical section
                await self.lock.acquire()
                chunk_dict = self.process_chunking('cfn')
                self.lock.release()
                for chunk_idx, byte in chunk_dict.items():
                    await websocket.send(byte)
                await websocket.send('end')

            # if message == 'mv2_new':
            #     print('entered chunking loop')
            #
            #     #critical section
            #     await self.lock.acquire()
            #     chunk_dict = self.process_chunking('mv2_new')
            #     self.lock.release()
            #
            #     tasks = []
            #     for chunk_idx , byte in chunk_dict.items():
            #         await websocket.send(byte)
            #     await websocket.send('end')


            else:
                if isinstance(message,bytes):
                    self.agg = True
                    counter = 0
                    recv_bytes = message
                    while True:
                        if recv_bytes == 'end':
                            break
                        else:
                            print('This is the size of the {} byte : {}'.format(counter, sys.getsizeof(recv_bytes)))
                            self.modelBytes[counter] = recv_bytes
                            counter += 1
                        recv_bytes = await self.broadcast_queue.get()

                if self.nextMsg and self.agg:
                    # if self.msgInfo == 'mv2':
                    #     print('aggregating model weights.....')
                    #     #critical section
                    #     await self.lock.acquire()
                    #     response = self.aggregation_cs()
                    #     self.lock.release()
                    if self.msgInfo == 'mv1':
                        print('Aggregating MobilenetV1')
                        # critical section
                        await self.lock.acquire()
                        response = self.aggregation_cs(self.msgInfo)
                        self.lock.release()
                    elif self.msgInfo == 'cfn':
                        print('Aggregating Cifarnet')
                        # critical section
                        await self.lock.acquire()
                        response = self.aggregation_cs(self.msgInfo)
                        self.lock.release()

                else:
                    #process the message
                    response = self.process_message(message)
                #send the response
                await websocket.send(response)

            self.connected_clients.remove(websocket)


    def process_chunking(self,model_type):

        print('in chunk handling procedure')
        # if model_type == 'mv2_new':
        #     PATH = './database/mobilenetv2_global_base.pth'
        #     if not os.path.isfile(PATH):
        #         model = get_mv2net()
        #         save_model(model,PATH)

        if model_type == 'mv1':
            PATH = './database/mobilenetv1_global.pth'
            if not os.path.isfile(PATH):
                model = get_mv1net()
                save_model(model,PATH)

        elif model_type == 'cfn':
            PATH = './database/cifarnet_global.pth'
            if not os.path.isfile(PATH):
                model = get_cifarnet()
                save_model(model,PATH)

        with open(PATH, "rb") as model_data:
            byte_data = model_data.read()

        chunking_split = 2**19
        chunk_dict = {}
        counter = 0
        for i in range(0,len(byte_data),chunking_split):
            print('This is the size of the {} byte : {}'.format(counter, sys.getsizeof(byte_data[i:i+chunking_split])))
            chunk_dict[counter] = byte_data[i:i+chunking_split]
            counter += 1
        print(len(chunk_dict))
        return chunk_dict




    def process_message(self,message):
        '''
        Websocket only send str, so if the data is a dict obj,
        it will send only the string key, thats why we need
        to serialize the data. using pickle works since json
        cannot serialise tensor.
        '''



        if message == "Hello":
            return "World!"

        elif message == 'aggregate_mv2' and self.nextMsg == False: #assume is pytorch model first
            self.nextMsg = True
            self.msgInfo = 'mv2'
            print('aggregate info received')
            return 'aggregate recv'

        elif message == 'aggregate_mv1' and self.nextMsg == False: #assume is pytorch model first
            self.nextMsg = True
            self.msgInfo = 'mv1'
            print('aggregate info received')
            return 'aggregate recv'
        elif message == 'aggregate_cifarnet' and self.nextMsg == False: #assume is pytorch model first
            self.nextMsg = True
            self.msgInfo = 'cfn'
            print('aggregate info received')
            return 'aggregate recv'

        else:
            print(message)
            print(type(message))
            print("ERROR")

    def aggregation_cs(self,model_type):

        if model_type == 'mv1':
            PATH = "./database/mobilenetv1_recv.pth"
            Main_PATH = "./database/mobilenetv1_global.pth"
            model_temp = get_mv1net()
            model_main = get_mv1net()
        elif model_type == 'cfn':
            PATH = "./database/cifarnet_recv.pth"
            Main_PATH = "./database/cifarnet_global.pth"
            model_temp = get_cifarnet()
            model_main = get_cifarnet()


        with open(PATH, 'wb') as model_data:
            for chunk_idx, model_bytes in self.modelBytes.items():
                model_data.write(model_bytes)

        # average the weights (Needs some error checking)
        model_temp.load_state_dict(torch.load(PATH))
        if os.path.isfile(Main_PATH):
            model_main.load_state_dict(torch.load(Main_PATH))

        sdTemp = model_temp.state_dict()
        sdMain = model_main.state_dict()

        for key in sdTemp:
            sdMain[key] = (sdMain[key] + sdTemp[key]) / 2

        # Check that can load into the model
        model_main.load_state_dict(sdMain)

        # save new model
        torch.save(model_main.state_dict(), Main_PATH)

        self.nextMsg = False
        self.agg = False
        return 'Aggregation Done! no errors found!'

    async def _handler(self,websocket, path):
        """Setup the consumer and producer response handlers with asyncio.
           Args:
                websocket: the websocket connection to the client
        """

        asyncio.set_event_loop(self.loop)
        consumer_task = asyncio.ensure_future(self._consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self._producer_handler(websocket))

        done,pending = await asyncio.wait(
            [consumer_task,producer_task],return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

    def start(self):
        print("Starting the server...")
        start_server = websockets.serve(self._handler, self.host, self.port)


        asyncio.get_event_loop().run_until_complete(start_server)
        print("Server started...")
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            print("Websocket server stopped...")
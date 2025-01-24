import MetaTrader5 as mt5
import pandas as pd 
import numpy as np
import pickle
import torch
from torch import nn
from functools import reduce
from torch.serialization import add_safe_globals
import time
import datetime
import threading as th
import pyautogui

class Bot:

    def __init__(self , login , pas , server ,  path  ) : 
        self.login = login 
        self.pas = pas
        self.server = server 
        self.path = path 

    def init(self):

        init = mt5.initialize(path=self.path)
        login = mt5.login(self.login, password=self.pas , server=self.server)
        return init , login
    
    def get_data(self , symbol , timeframe = mt5.TIMEFRAME_M5 , n = None ):
        if n == None :
            n = 600
        # symbol = self.symbol
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')
        df['time'] = df['time'].dt.tz_localize(None)
        # df.set_index('time', inplace=True)
 

        return df

    def prepross(self , df):
       
        df.drop(['tick_volume' , 'spread' , 'real_volume'] , inplace=True , axis=1)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        df15 = df.resample('15T').agg({
            'open': 'first',    # قیمت شروع بازه زمانی
            # 'high': 'max',      # بالاترین قیمت
            # 'low': 'min',       # پایین‌ترین قیمت
            'close': 'last',    # حجم معامله
        })

        df30 = df.resample('30T').agg({
            'open': 'first',    # قیمت شروع بازه زمانی
            # 'high': 'max',      # بالاترین قیمت
            # 'low': 'min',       # پایین‌ترین قیمت
            'close': 'last',    # حجم معامله
        })

        df1h = df.resample('H').agg({
            'open': 'first',    # قیمت شروع بازه زمانی
            # 'high': 'max',      # بالاترین قیمت
            # 'low': 'min',       # پایین‌ترین قیمت
            'close': 'last',    # حجم معامله
        })

        df4h = df.resample('4H').agg({
            'open': 'first',    # قیمت شروع بازه زمانی
            # 'high': 'max',      # بالاترین قیمت
            # 'low': 'min',       # پایین‌ترین قیمت
            'close': 'last',    # حجم معامله
        })

        df15.rename(columns={'open': 'open15',  'close': 'close15'}, inplace=True)
        df30.rename(columns={'open': 'open30',  'close': 'close30'}, inplace=True)
        df1h.rename(columns={'open': 'open1h',  'close': 'close1h'}, inplace=True)
        df4h.rename(columns={'open': 'open4h',  'close': 'close4h'}, inplace=True)

        main = [df , df15  , df30 , df1h , df4h]
        df = reduce(lambda left, right: pd.merge(left, right, on='time', how='outer'), main)

        df['5']  = np.where(df['close'] - df['open'] > 0 , 1 , -1)
        df['10'] = np.where(df['close'] - df['open'] > 0 , 1 , -1)
        df['15'] = np.where(df['close'] - df['open'] > 0 , 1 , -1)
        df['30'] = np.where(df['close'] - df['open'] > 0 , 1 , -1)
        df['H1'] = np.where(df['close'] - df['open'] > 0 , 1 , -1)
        df['H4'] = np.where(df['close'] - df['open'] > 0 , 1 , -1)

        df.drop(['open' , 'close' , 'high' , 'low' , 'open15' , 'close15' , 'open30' , 'close30' , 'open1h' , 'close1h' ,
                'open4h' , 'close4h' ] , inplace = True , axis = 1)
        
        df.reset_index(inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df.drop('time',axis=1,inplace=True)
        

        return df

    def scale(self , df , scale):
        df_scale = scale.transform(np.array(df))
        return df_scale
    
    def tensor(self , df):
        tensor = torch.FloatTensor(df)
        return tensor[0].unsqueeze(0)


class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, bidirectional, num_cls, batch_first ):
      super().__init__()
      self.rnn = nn.LSTM(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          bidirectional=bidirectional,
          batch_first=batch_first ,
      )
      self.fc = nn.LazyLinear(num_cls)


  def forward(self, x):
      outputs, (hn, cn) = self.rnn(x)  # خروجی‌ها و hidden stateها
      # استفاده از آخرین hidden state (آخرین گام زمانی)
    #   last_output = outputs[:, -1, :]  # شکل: (batch_size, hidden_size)
      y = self.fc(outputs)
      y = torch.sigmoid(y)
      return y

 
def pred(model , x):
    model.eval()
    with torch.no_grad():
        return model(x).item()



def trade(type):
    if type == 'buy' : 

        pyautogui.keyDown('shift')  # نگه داشتن کلید Shift
        pyautogui.press('w')        # فشار دادن کلید W
        time.sleep(0.1)
        pyautogui.keyUp('shift')
        pyautogui.keyUp('w')
        


    else : 
        pyautogui.keyDown('shift')  # نگه داشتن کلید Shift
        pyautogui.press('s')        # فشار دادن کلید W
        time.sleep(0.1)
        pyautogui.keyUp('shift')
        pyautogui.keyUp('s')
        

def run():
    while True :
        time.sleep(1)
        if (datetime.datetime.now().minute + 1) % 5 == 0 and (datetime.datetime.now().second + 1) >= 58 : 
            
            df = trade.get_data(symol)
            df = trade.prepross(df)
            df = df.iloc[-500 : ]
            df_scale = trade.scale(df , scaler)
            df_main = trade.tensor(df_scale)
            pred = pred(model , df_main)
            if pred > 0.95 or pred < 0.05 : 
                trade('buy' if pred > 0.95 else 'sell')
                


add_safe_globals([LSTM])
model = torch.load('best_model_lstm.pth', weights_only=False)


login = 225762
pas = 'z$Fh{}3z'
server = 'PropridgeCapitalMarkets-Server'
path = r"C:\Program Files\Propridge Capital Markets MT5 Terminal\terminal64.exe"
symol = 'US30'

with open('r.pkl', 'rb') as f:
    scaler = pickle.load(f)



trade = Bot(login = login , pas = pas , server= server , path = path  )
trade.init()


th.Thread(target= run).start()










import schedule 
import time 
import sqlite3

import requests
import json


#___________________________________________________________________________________________________________________
# scheduler database
# A defferent python code runnig for cheking training status and train data
# for that scheduler program (scheduler.py) we need to set a status code
# for setting status a new sqlite db is created
# This DB contains only one table - scheduler
# scheduler table attributes are - ID, start_training, num_iter
# ID - primary key
# start_training - Boolean field - for checking training status
# num_iter - number of itrations for training data
#___________________________________________________________________________________________________________________
print("-" * 60)
#______________________________________________________________________________________________________
# scheduler for training
#______________________________________________________________________________________________________
conn = sqlite3.connect('scheduler.db')
print ("scheduler database Opened successfully")
class SqliteSchedulerDB:
    def create_db(self):
        
        conn.execute('''CREATE TABLE scheduler
                (ID INT PRIMARY KEY     NOT NULL,
                start_training           BOOLEAN,
                is_in_progress           BOOLEAN,
                num_iter INT);''')
        print ("Table created successfully")
    def insert_value(self):
        conn.execute("INSERT INTO scheduler (ID,start_training,is_in_progress,num_iter) \
            VALUES (1, 0, 0, 100)")

        conn.commit()
        print ("Records created successfully")
    def update_training(self, status, num_iter = 1000,conn=conn):        
        conn.execute("UPDATE scheduler set start_training = %d, num_iter = %d where ID = 1"%(status, num_iter))
        conn.commit()
        print ("Total number of rows updated :", conn.total_changes)
    def update_is_in_progress(self, status,conn=conn):        
        conn.execute("UPDATE scheduler set is_in_progress = %d where ID = 1"%(status))
        conn.commit()
        print ("Total number of rows updated :", conn.total_changes)
    
    def check_status(self, conn=conn):
        cursor = conn.execute("SELECT start_training,is_in_progress, num_iter  from scheduler")
        for row in cursor:            
            print ("start_training = ", row[0])   
            print ("is_in_progress = ", row[1])            
            print ("num_iter = ", row[2]) 
        return row

obj_SqliteSchedulerDB = SqliteSchedulerDB()
try:       
    obj_SqliteSchedulerDB.check_status(conn=conn)
except:
    obj_SqliteSchedulerDB.create_db()
    obj_SqliteSchedulerDB.insert_value()
conn.close()  

print("-" * 60)     
#___________________________________________________________________________________________________________________

# ___________________________________________________________________________________________________________________
# sheduler for training data. it will execute every 1 min interverl
# ___________________________________________________________________________________________________________________
BASE_URL = 'http://192.168.0.231:2424/'
# BASE_URL = 'http://127.0.0.1:8000/'
ENDPOINT = 'hard_train/'

def training_data():
    print("-" * 50)
    print("scheduler...")
    conn = sqlite3.connect('scheduler.db')
    row = obj_SqliteSchedulerDB.check_status(conn=conn)
    conn.close()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print('current_time', current_time)
    if not row[1]:
        if row[0]:
            context = {
                'train' : True,
                'n_iter' : row[2] # optional parameter. Default value is 100
            }
            
            try: 
                resp = requests.post(BASE_URL+ENDPOINT, data=context)
            except:
                resp = None
                
            if resp:
                print(resp.status_code)
                if resp.status_code == 200:
                    print(resp.json())
                if resp.status_code == 406:
                    print(resp.json())
                if resp.status_code == 400:
                    print(resp.json())
            else:
                print("can't connect to server")
        else:
            print("no training request found.")
    else:
        print("In progress")

schedule.every(0.2).minutes.do(training_data) 

while True: 
    schedule.run_pending() 
    time.sleep(1) 
from __future__ import unicode_literals, print_function
from django.shortcuts import render
from django.views.generic import View
from django.http import HttpResponse
import json
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from retrain_fer.models import QueryDB, EntityDB
# training

import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm # loading bar

import os
import sqlite3

from nltk.tokenize import sent_tokenize
#custom
from retrain_fer.mixin import SerializerMixin, HttpresponseMixin  
from retrain_fer.forms import QueryDBForm, EntityDBForm
from retrain_fer.utils import is_json

import pdb
from pprint import pprint

LABEL = 'ORG'

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
            VALUES (1, 0, 0, 100)");

        conn.commit()
        print ("Records created successfully")
    def update_training(self, status, num_iter = 100,conn=conn):        
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
#======================================================================================================
# REST API
#______________________________________________________________________________________________________
# Save Training data to db
# ========================
# table 1: QueryDB
# attributes: id, query
# table 2: EntityDB
# attributes: id, query(foregn field:- QueryDB),entity_name, start_pos, end_pos
# input format
# ============
# {
#   'query' : <query :: str>,
#   'entity_list' : [(<start pos :: int>, <end pos :: int>, <entity :: str>)]
# }
# eg:
# {
#   'query' :"Horses are too tall and horses are pretend to care about your feelings",
#   'entity_list' : [(0, 6, 'ANIMAL'), (25, 31, 'ANIMAL')]
# }
# output
# ======
# {
#   'result' : result,
#   'error' : error
# }
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class FerSaveData2DB(HttpresponseMixin, SerializerMixin, View):
    def get_object_by_query(self,query):
        try:
            emp = QueryDB.objects.get(query=query)
        except QueryDB.DoesNotExist:
            emp = None
        return emp
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs):
        result = []
        error = []   
        flag = True

        data = request.body
        valid_jason = is_json(data)
        if valid_jason:
            data = json.loads(data)
            if 'query' in data:
                query = data['query']
            else:
                error.append('query is not given')
                flag = False
                status = 400
            
            if 'entity_list' in data:
                entity_list = data['entity_list']
            else:
                error.append('entity is not given')
                flag = False
                status = 400
        else:
            if 'query' in request.POST:
                query = request.POST['query']
            else:
                error.append('query is not given')
                flag = False
                status = 400
            
            if 'entity_list' in request.POST:
                entity_list = request.POST.getlist('entity_list')
            else:
                error.append('entity is not given')
                flag = False
                status = 400        
        if flag:            
            query_data = {                
                'query' : query
            } 
            query_form = QueryDBForm(query_data)
            
            if query_form.is_valid():
                query_form.save(commit=True)                
                
            if query_form.errors:
                error.append(query_form.errors)
                status = 400
            # save entity
            
            for i in range(0,len(entity_list),2):
                # pdb.set_trace()
                
                entity_data = {                    
                    'query' : query,
                    'entity_name' : 'ORG', #entity_list[i+2],
                    'start_pos' : int(entity_list[i+0]),
                    'end_pos' : int(entity_list[i+1]),
                    'is_trained' : False
                } 
                entity_form = EntityDBForm(entity_data)
                if entity_form.is_valid():                    
                    entity_form.save(commit=True)
                    
                    # finding id of saved data
                    obj_cur_data = EntityDB.objects.filter(query=query,start_pos=int(entity_list[i+0]),end_pos=int(entity_list[i+1]), is_trained = False).order_by('-id')  
                    for obj in obj_cur_data:
                        tmp_entity_data = {}
                        tmp_entity_data['id'] = obj.id
                        tmp_entity_data['query'] = query                        
                        tmp_entity_data['start_pos'] = int(entity_list[i+0])
                        tmp_entity_data['end_pos'] = int(entity_list[i+1])
                        result.append(tmp_entity_data)  
                    status = 200
                if entity_form.errors:
                    error.append(entity_form.errors)
                    status = 400
        f_result = {
            'result' : result,
            'error' : error
        }   
        json_data = json.dumps(f_result)
        return self.render_to_http_response(json_data, status=status)
#______________________________________________________________________________________________________
# new entity label
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def Train(TRAIN_DATA, model=None, new_model_name='ORG', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""    
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    conn = sqlite3.connect('scheduler.db')
    obj_SqliteSchedulerDB.update_is_in_progress(status=1, conn=conn)
    
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in tqdm(TRAIN_DATA):
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # update training status to False. 
        # The scheduler will not call training data function        
        obj_SqliteSchedulerDB.update_training(status=0, conn=conn) 
        obj_SqliteSchedulerDB.update_is_in_progress(status=0, conn=conn)
        conn.close()       
        # # test the saved model
        # print("Loading from", output_dir)
        # nlp2 = spacy.load(output_dir)
        # doc2 = nlp2(test_text)
        # for ent in doc2.ents:
        #     print(ent.label_, ent.text)
def Test(test_text, model=None):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model     
    else:
        nlp = spacy.blank('en')  # create blank Language class
        # print("Created blank 'en' model")
    # test the trained model
    # sent_token = sent_tokenize(test_text) 
    final_res = []
       
    # for sent in sent_token:
    result = {}
    doc = nlp(test_text)
    result["sentence"]  = test_text
    res = []        
    for ent in doc.ents:          
        res.append({"type" :ent.label_,
        "value" : ent.text}) 
    result["entity"] = res
    final_res.append(result)
    return final_res

#______________________________________________________________________________________________________
# Training data
# =============
# input 
# =====
# ENDPOINT = 'train/'
# context = {
#     'train' : True,
#     'n_iter' : 100 # optional parameter. Default value is 100
# }
# output
# ======
# {'result': ['Training started'], 
# 'error': []}
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class TrainData(HttpresponseMixin, SerializerMixin, View):    
    def assure_path_exists(self, path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
            flag = False
        else:
            flag = True
        return flag
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs):
        result = []
        error = []
        n_iter = 100
        flag = train = False
        if 'train' in request.POST:
            train = request.POST['train']  
        else:
            flag = False
            error.append("train is not given")          
        if 'n_iter' in request.POST:
            n_iter = int(request.POST['n_iter'])
        if train == "True" or train == 1:            
            # if status of training is set to in progress then the scheduler execute training function
            # update scheduler is_trained status to True.
            conn = sqlite3.connect('scheduler.db')
            obj_SqliteSchedulerDB.update_training(status=1, num_iter=n_iter, conn=conn)            
            conn.close()
            # train(TRAIN_DATA = TRAIN_DATA, model=model, new_model_name='ORG', output_dir=output_dir, n_iter=n_iter)
            result.append("Training started")
        else:
            error.append("Please set value of train to True for start training entity")
        json_data = json.dumps({'result': result, 'error' : error})
        return HttpResponse(json_data, content_type='application/json')
#______________________________________________________________________________________________________
# HardTrainData
# This is for scheduler to start training data
# input 
# =====
# ENDPOINT = 'hard_train/'
# context = {
#     'train' : True,
#     'n_iter' : 100 # optional parameter. Default value is 100
# }
# output
# ======
# {'result': ['Training completed'], 
#  'error': []}
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class HardTrainData(HttpresponseMixin, SerializerMixin, View):
    def update_training_status(self, obj_entity, status):
        entity_data = {  
            'id' : obj_entity.id,                  
            'query' : obj_entity.query,
            'entity_name' : 'ORG', #entity_list[i+2],
            'start_pos' : obj_entity.start_pos,
            'end_pos' : obj_entity.end_pos,
            'is_trained' : status
        } 
        entity_form = EntityDBForm(entity_data, instance = obj_entity)
        if entity_form.is_valid():
            entity_form.save(commit=True)
            flag = True
        if entity_form.errors:
            flag = False
        return flag
    def assure_path_exists(self, path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
            flag = False
        else:
            flag = True
        return flag
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs):
        result = []
        error = []
        n_iter = 100
        flag = False
        if 'train' in request.POST:
            train = request.POST['train']  
        else:
            flag = False
            error.append("train is not given")          
        if 'n_iter' in request.POST:
            n_iter = int(request.POST['n_iter'])
        if train == "True" or train == '1':
            # TRAIN_DATA creation
            obj_EntityDB = EntityDB.objects.all()
            TRAIN_DATA = []
            for obj_ent in obj_EntityDB:                       
                TRAIN_DATA.append((obj_ent.query.query,{'entities' : [(obj_ent.start_pos, obj_ent.end_pos, obj_ent.entity_name)]}))
                # change training status
                if self.update_training_status(obj_entity = obj_ent, status=True):
                    print("can't update training status of", obj_ent.query.query, (obj_ent.start_pos, obj_ent.end_pos))
            output_dir = 'model/'        
            model = 'model/'            
            m_flag = self.assure_path_exists(path = model)
            if not m_flag:
                model = None   
            
            Train(TRAIN_DATA = TRAIN_DATA, model=model, new_model_name='ORG', output_dir=output_dir, n_iter=n_iter)
            result.append("Training completed")
        else:
            error.append("Please set value of train to True for start training entity")
        json_data = json.dumps({'result': result, 'error' : error})
        return HttpResponse(json_data, content_type='application/json')

#______________________________________________________________________________________________________
# Test data
# =============
# input 
# =====
# ENDPOINT = 'test/'
# context = {
#     'test_text' : "Do you know Zerone-consulting?"
# }
# output
# ======
# {'result': {'test_text': 'Do you know Zerone-consulting?', 
#             'entities': [['ORG', 'Zerone-consulting']]}, 
#  'error': []}
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class TestData(HttpresponseMixin, SerializerMixin, View):        
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs): 
        flag = False    
        error = []  
        result = {}
        n_iter = 100
        if 'test_text' in request.POST:
            test_text = request.POST['test_text']
            flag = True
        else:
            flag = False
            error.append("test_text is not given")
        if flag:
            model = 'model/'
            dir = os.path.dirname(model)
            if not os.path.exists(dir):
                model = None                     
            final_res_list = Test(test_text = test_text, model=model)
                        
        json_data = json.dumps({'result':final_res_list,
                        'error' : error})
        return HttpResponse(json_data, content_type='application/json')
#______________________________________________________________________________________________________

#______________________________________________________________________________________________________
# AllDatasinDB
# =============
# input 
# =====
# ENDPOINT = 'all_data/'
# context = {     
# }
# output
# ======
# {<EntityDB datas sorted descending>}, 
#  'error': []}
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class AllDatasinDB(HttpresponseMixin, SerializerMixin, View):        
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs):  
        # pdb.set_trace()
        if 'is_trained' in request.POST and request.POST['is_trained'] == '0':           
            all_datas = EntityDB.objects.filter(is_trained = False).order_by('-id') 
        else:
            all_datas = EntityDB.objects.all().order_by('-id') 
        json_data = SerializerMixin.serialize('json', all_datas)                       
        return HttpResponse(json_data, content_type='application/json')
#______________________________________________________________________________________________________

#______________________________________________________________________________________________________
# DeleteDatasinDB
# =============
# input 
# =====
# ENDPOINT = 'delete_data/'
# context = {
#     'id' : <id of entity_db>
# }
# output
# ======
# {'result': {"<id> deleted successfully"}, 
#  'error': []}
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class DeleteDatasinDB(HttpresponseMixin, SerializerMixin, View): 
    def get_object_by_id(self,id):
        try:
            emp = EntityDB.objects.get(id=id)
        except EntityDB.DoesNotExist:
            emp = None
        return emp
       
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs): 
        flag = False    
        error = []  
        result = [] 
        status = 200       
        if 'id' in request.POST:
            id = int(request.POST['id'])
            flag = True
        else:
            flag = False
            error.append("id is not given")
        if flag:
            entity = self.get_object_by_id(id)
            if entity is None:
                error.append("No matched resources are found for deletion")
                status=404
            else:
                status,deleted_item = entity.delete()
            
                if status == 1:
                    result.append("%d deleted successfully"%id)
                    status = 200
                else:
                    error.append("Unable to delete. Please try again")
                    status=404    
        json_data = json.dumps({'result':result,
                        'error' : error})
        return HttpResponse(json_data, content_type='application/json', status=status)
#______________________________________________________________________________________________________

#______________________________________________________________________________________________________
# UpdateDatasinDB
# =============
# input 
# =====
# ENDPOINT = 'update_data/'
# context = {
#     'id' : "<id>",
#     'query' : "<query>" # optional arg
#     'start_pos' : <start_pos> # optional arg
#     'end_pos' : <end_pos> # optional arg
# }
# output
# ======
# {'result': ["<id updated successfully>"], 
#  'error': []}
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class UpdateDatasinDB(HttpresponseMixin, SerializerMixin, View): 
    def get_object_by_id(self,id):
        try:
            emp = EntityDB.objects.get(id=id)
        except EntityDB.DoesNotExist:
            emp = None
        return emp       
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs): 
        flag = False    
        error = []  
        result = [] 
        status = 200       
        if 'id' in request.POST:
            id = int(request.POST['id'])
            flag = True
        else:
            flag = False
            error.append("id is not given")        
        if flag:
            entity = self.get_object_by_id(id)
            if entity is None:
                error.append("No matched resources are found for deletion")
                status=404
                flag = False            
            
            else:
                if 'query' in request.POST:            
                    query = request.POST['query'] 
                    query_data = {                
                        'query' : query
                    } 
                    query_form = QueryDBForm(query_data)
                    # pdb.set_trace()
                    if query_form.is_valid():
                        query_form.save(commit=True)                
                        flag = True
                    if query_form.errors:
                        error.append(query_form.errors)
                        status = 400
                        flag = False
                else:
                    query =  entity.query                          
                if 'start_pos' in request.POST:
                    start_pos = int(request.POST['start_pos'])
                else:
                   start_pos = entity.start_pos
                if 'end_pos' in request.POST:
                    end_pos = int(request.POST['end_pos'])
                else:
                    end_pos = entity.end_pos
                if flag:
                    entity_data = {  
                        'id' : id,                  
                        'query' : query,
                        'entity_name' : 'ORG', #entity_list[i+2],
                        'start_pos' : start_pos,
                        'end_pos' : end_pos,
                        'is_trained' : False
                    } 
                    entity_form = EntityDBForm(entity_data, instance = entity)
                    if entity_form.is_valid():
                        entity_form.save(commit=True)
                        result.append('%d updated successfully'%(id))                      
                        status = 200
                    if entity_form.errors:
                        error.append(entity_form.errors)
                        status = 400                              
        json_data = json.dumps({'result':result,
                        'error' : error})
        return HttpResponse(json_data, content_type='application/json', status=status)
#______________________________________________________________________________________________________
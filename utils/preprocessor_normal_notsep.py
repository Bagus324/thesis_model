import os
import sys
import pickle
import re
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


import pandas as pd
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl

from transformers import AutoTokenizer

class Preprocessor(pl.LightningDataModule):

    def __init__(self, max_length = 100, batch_size = 32):
        super(Preprocessor, self).__init__()
        torch.manual_seed(1)
        self.tokenizers = AutoTokenizer.from_pretrained('FacebookAI/xlm-mlm-100-1280')
        # self.tokenizers = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
        self.max_length = max_length
        self.batch_size = batch_size
        self.dataset_type = "wps"
        self.sastrawi = True
        self.pickle_file_path = '/kaggle/working/twitter_sentimen/skripsi/preprocessed/sastrawi_undersam.pkl'
        if self.sastrawi:
            self.stemmer = StemmerFactory().create_stemmer()

    def clean_str(self, string):
        string = re.sub(r"\bRT\b", " ", string)
        string = re.sub(r"\bUSER\b", " ", string)
        string = string.lower()
        string = re.sub(r"\brt\b", " ", string)
        string = re.sub(r"\buser\b", " ", string)
        string = re.sub(r"[^A-Za-z0-9]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        # string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"\'t", " ", string)
        string = re.sub(r"\n", "", string)
        # string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        # string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", "", string)
        string = re.sub(r"!", "", string)
        string = re.sub(r"\(", "", string)
        string = re.sub(r"\)", "", string)
        string = re.sub(r"\?", "", string)
        # string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()

        return string
    
    def stemmer_text(self, string):
        string = self.stemmer.stem(string)
        return string

    def load_data(self, type):
        if type == "wops":
            data = pd.read_csv('datasets/binary.csv', sep=",", encoding="ISO-8859-1")
            data = data.dropna(how="any")
            condition_empty_label = data[
                (
                    (data['HS'] == 0) &
                    (data['Abusive'] == 0) &
                    (data['HS_Individual'] == 0) &
                    (data['HS_Group'] == 0) &
                    (data['HS_Religion'] == 0) &
                    (data['HS_Race'] == 0) &
                    (data['HS_Physical'] == 0) &
                    (data['HS_Gender'] == 0) &
                    (data['HS_Other'] == 0) &
                    (data['HS_Weak'] == 0) &
                    (data['HS_Moderate'] == 0) &
                    (data['HS_Strong'] == 0)
                )
            ].index
            data = data.drop(condition_empty_label)

            tweet = data
            tweet["Tweet"] = tweet["Tweet"].apply(lambda x: self.clean_str(x))
            tweet = tweet
            separated = self.sepping(tweet)
            separated = separated
        elif type == "wps":
            if self.sastrawi == True and os.path.exists(self.pickle_file_path):
                # Load the data from the existing pickle file
                print("==========PREPROCESSED DATA FOUND==========")
                with open(self.pickle_file_path, 'rb') as pickle_file:
                    tweet = pickle.load(pickle_file)
                tqdm.pandas(desc="==========CLEANING==========")
                
                tweet["Tweet"] = tweet["Tweet"].progress_apply(lambda x: self.clean_str(x))
                print(f"Loaded existing data: {tweet}")
            else:
                data = pd.read_csv('/kaggle/working/twitter_sentimen/skripsi/datasets/df_with_positif.csv', sep=",", encoding="ISO-8859-1")
                data = data.dropna(how="any")
                tweet = data
                tweet["Tweet"] = tweet["Tweet"].apply(lambda x: self.clean_str(x))
                tweet = tweet.sample(frac=1)
                separated = self.sepping_wps(tweet)
                separated = separated
            # separated = self.sepping_wps(tweet)
            # separated = separated
        # data_ite = pd.read_csv('datasets/Dataset Twitter Fix - Indonesian Sentiment Twitter Dataset Labeled (1).csv', sep=",", encoding="ISO-8859-1")
        # data = self.converter(data_ite, data)
        # Semua kolom di cek yang kosong di drop/
        
        # Spesifik remove karakter null berdasarkan kolom
        # data = data[data.notna()]
        
        
        # label = []
        # X_data = []
        # for line in separated:
        #     temp = [int(i) for i in line[1:]]
        #     label.append(temp)
        #     X_data.append(line[:1])
        

        # self.labels = tweet.columns.tolist()[1:]



        # ### MORE SPLITTING DATASET ### INTO TRAIN-VAL AND TEST FROM WHOLE ###

        # X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, label, test_size=0.2, random_state=1, stratify=label)

        # ### MORE SPLITTIN DATASET ### INTO TRAIN AND VALIDATION FROM TRAIN-VAL ###
        # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=1, stratify=y_train_val)

        # x_input_ids_train, x_token_type_ids_train, x_attention_mask_train = [], [], []
        # x_input_ids_val, x_token_type_ids_val, x_attention_mask_val = [], [], []
        # x_input_ids_test, x_token_type_ids_test, x_attention_mask_test = [], [], []

        # ### TOKENIZING TRAIN ###
        
        # for tw in X_train:
        #     tkn_tweet = self.tokenizers(text = tw,
        #                                 max_length = self.max_length,
        #                                 padding = 'max_length',
        #                                 truncation = True)
        #     x_input_ids_train.append(tkn_tweet['input_ids'][0])
        #     x_token_type_ids_train.append(tkn_tweet['token_type_ids'][0])
        #     x_attention_mask_train.append(tkn_tweet['attention_mask'][0])
        
        # x_input_ids_train = torch.tensor(x_input_ids_train)
        # x_token_type_ids_train = torch.tensor(x_token_type_ids_train)
        # x_attention_mask_train = torch.tensor(x_attention_mask_train)
        # y_train = torch.tensor(y_train)

        # train_dataset = TensorDataset(x_input_ids_train, x_token_type_ids_train, x_attention_mask_train, y_train)

        # ### TOKENIZING VALIDATION ###

        # for tw in X_val:
        #     tkn_tweet = self.tokenizers(text = tw,
        #                                 max_length = self.max_length,
        #                                 padding = 'max_length',
        #                                 truncation = True)

        #     x_input_ids_val.append(tkn_tweet['input_ids'][0])
        #     x_token_type_ids_val.append(tkn_tweet['token_type_ids'][0])
        #     x_attention_mask_val.append(tkn_tweet['attention_mask'][0])
        
        # x_input_ids_val = torch.tensor(x_input_ids_val)
        # x_token_type_ids_val = torch.tensor(x_token_type_ids_val)
        # x_attention_mask_val = torch.tensor(x_attention_mask_val)
        # y_val = torch.tensor(y_val)

        # valid_dataset = TensorDataset(x_input_ids_val, x_token_type_ids_val, x_attention_mask_val, y_val)

        # ### TOKENIZING TEST ###

        # for tw in X_test:
        #     tkn_tweet = self.tokenizers(text = tw,
        #                                 max_length = self.max_length,
        #                                 padding = 'max_length',
        #                                 truncation = True)

        #     x_input_ids_test.append(tkn_tweet['input_ids'][0])
        #     x_token_type_ids_test.append(tkn_tweet['token_type_ids'][0])
        #     x_attention_mask_test.append(tkn_tweet['attention_mask'][0])
        
        # x_input_ids_test = torch.tensor(x_input_ids_test)
        # x_token_type_ids_test = torch.tensor(x_token_type_ids_test)
        # x_attention_mask_test = torch.tensor(x_attention_mask_test)
        # y_test = torch.tensor(y_test)

        # test_dataset = TensorDataset(x_input_ids_test, x_token_type_ids_test, x_attention_mask_test, y_test)
        # torch.save(test_dataset, '/kaggle/working/twitter_sentimen/skripsi/preprocessed/test_dm.pt')

        
        # separated = separated[:50]
        label, X_data = [], []
        for line in tweet.values.tolist():
            X_data.append(line[:1])
            label.append(line[1:])

        self.labels = tweet.columns.tolist()[1:]
        print("==================", self.labels)

        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

        for tw in X_data:
            # text = ' '.join(map(str, tw[:1]))  #langsung diambil tweetnya
            tkn_tweet = self.tokenizers(text = tw,
                                        max_length = self.max_length,
                                        padding = 'max_length',
                                        truncation = True)

            x_input_ids.append(tkn_tweet['input_ids'][0])
            x_token_type_ids.append(tkn_tweet['token_type_ids'][0])
            x_attention_mask.append(tkn_tweet['attention_mask'][0])


        

        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(label)

        
        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids,x_attention_mask, y)
        # Standard
        # 80% (Training validation) 20% (testing)
        # training = 90% validation = 10%

        train_valid_dataset, test_dataset = torch.utils.data.random_split(
            tensor_dataset, [
                round(len(tensor_dataset) * 0.8), 
                round(len(tensor_dataset) * 0.2)
            ]
        )

        train_len = round(len(train_valid_dataset) * 0.9)
        valid_len = len(train_valid_dataset) - round(len(train_valid_dataset) * 0.9)

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_valid_dataset,
            [train_len, valid_len]
        )
        # print(len(train_dataset))
        # print(len(test_dataset))
        # print(len(valid_dataset))
        # sys.exit()

        return train_dataset, valid_dataset, test_dataset



    def setup(self, stage = None):
        train_data, valid_data, test_data = self.load_data(self.dataset_type)
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data

    def train_dataloader(self):
        sampler = RandomSampler(self.train_data)
        return DataLoader(
            dataset = self.train_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        sampler = SequentialSampler(self.valid_data) 
        return DataLoader(
            dataset = self.valid_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        sampler = SequentialSampler(self.test_data)
        return DataLoader(
            dataset = self.test_data,
            batch_size = self.batch_size,
            sampler = sampler,
            num_workers = multiprocessing.cpu_count()
        )

    def converter(self, origin, destination):
        x=1
        for i, sentimen in enumerate(origin["sentimen"]):
            if sentimen == 0:
                if x == 1:
                    list_1 = [origin["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    x+=1
                elif x == 2:
                    list_2 = [origin["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    destination = pd.concat([destination, a], ignore_index=True)
                    x=1
            elif sentimen == 1:
                if x == 1:
                    list_1 = [origin["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    x+=1
                elif x == 2:
                    list_2 = [origin["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    destination = pd.concat([destination, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 1:
                if x == 1:
                    list_1 = [origin["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    x+=1
                elif x == 2:
                    list_2 = [origin["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,0,0,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    destination = pd.concat([destination, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 2:
                if x == 1:
                    list_1 = [origin["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,1,0,0]
                    x+=1
                elif x == 2:
                    list_2 = [origin["Tweet"].loc[i],0,0,0,0,0,0,0,0,0,1,0,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    destination = pd.concat([destination, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 4:
                if x == 1:
                    list_1 = [origin["Tweet"].loc[i],1,1,1,0,0,0,1,1,0,0,1,0]
                    x+=1
                elif x == 2:
                    list_2 = [origin["Tweet"].loc[i],1,1,1,0,0,0,1,1,0,0,1,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    destination = pd.concat([destination, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 5:
                if x == 1:
                    list_1 = [origin["Tweet"].loc[i],1,1,1,0,0,0,0,1,0,0,1,0]
                    x+=1
                elif x == 2:
                    list_2 = [origin["Tweet"].loc[i],1,1,1,0,0,0,0,1,0,0,1,0]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    destination = pd.concat([destination, a], ignore_index=True)
                    x=1
#===============================================================================================
            elif sentimen == 6:
                if x == 1:
                    list_1 = [origin["Tweet"].loc[i],1,1,0,1,1,1,1,0,1,0,0,1]
                    x+=1
                elif x == 2:
                    list_2 = [origin["Tweet"].loc[i],1,1,0,1,1,1,1,0,1,0,0,1]
                    a = pd.DataFrame([[i for i in list_1], [i for i in list_2]],
                                        columns=["Tweet",
                                                "HS",
                                                "Abusive",
                                                "HS_Individual",
                                                "HS_Group",
                                                "HS_Religion",
                                                "HS_Race",
                                                "HS_Physical",
                                                "HS_Gender",
                                                "HS_Other",
                                                "HS_Weak",
                                                "HS_Moderate",
                                                "HS_Strong"])
                    destination = pd.concat([destination, a], ignore_index=True)
                    x=1
        return destination

    def sepping(self, data):
        
        final_data = []
        for line in data.values.tolist():
            label = line[1:]
            indexing = [i for i, l in enumerate(label) if l == 1]
            if len(indexing) >= 1:
                for i, isi in enumerate(label):
                    wrapper = np.zeros((12), dtype=int).tolist()
                    if isi == 1:
                        wrapper[i]=1
                        final_data.append([line[0]]+wrapper)
        
        return final_data
    
    def sepping_wps(self, data):
        
        final_data = []
        for line in data.values.tolist():
            label = line[1:]
            indexing = [i for i, l in enumerate(label) if l == 1]
            if len(indexing) >= 1:
                for i, isi in enumerate(label):
                    wrapper = np.zeros((13), dtype=int).tolist()
                    if isi == 1:
                        wrapper[i]=1
                        final_data.append([line[0]]+wrapper)
        
        return final_data

# if __name__ == '__main__':
#     pretox = PreprocessorToxic()
#     train_dataset, valid_dataset, test_dataset = pretox.load_data()
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import re

from utils.preprocessor_sastrawi import Preprocessor as Pre_Sastrawi
from utils.preprocessor_sastrawi_notsep import Preprocessor as Pre_Sastrawi_NotSeperated
from utils.preprocessor_normal_notsep import Preprocessor as Pre_Normal_NotSeperated
from utils.preprocessor_normal import Preprocessor as Pre_Normal

from model.xlm_skripsi import MultilabelXLMModel as XLM
from model.mbert_skripsi import Multilabelmbert as MBERT
# import argparse

# def clean_str(string):
#         string = string.lower()
#         string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
#         string = re.sub(r"\'s", " \'s", string)
#         string = re.sub(r"\'ve", " \'ve", string)
#         string = re.sub(r"n\'t", " n\'t", string)
#         string = re.sub(r"\n", "", string)
#         string = re.sub(r"\'re", " \'re", string)
#         string = re.sub(r"\'d", " \'d", string)
#         string = re.sub(r"\'ll", " \'ll", string)
#         string = re.sub(r",", " , ", string)
#         string = re.sub(r"!", " ! ", string)
#         string = re.sub(r"\(", " \( ", string)
#         string = re.sub(r"\)", " \) ", string)
#         string = re.sub(r"\?", " \? ", string)
#         string = re.sub(r"\s{2,}", " ", string)
#         string = string.strip()

#         return string

# def predictor(trainer, labels, threshold = 0.5):
#     trained_model = MultiLabelModel.load_from_checkpoint(
#         trainer.checkpoint_callback.best_model_path,
#         labels = labels
#     )
#     trained_model.eval()
#     trained_model.freeze()

#     tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

#     test_comment = "sih kerja delay mulu setan"
#     test_comment = clean_str(test_comment)
#     encoding = tokenizer.encode_plus(
#         test_comment,
#         add_special_tokens = True,
#         max_length = 100,
#         return_token_type_ids = True,
#         padding = "max_length",
#         return_attention_mask = True,
#         return_tensors = 'pt',
#     )

#     test_prediction = trained_model(encoding["input_ids"], 
#                                     encoding["token_type_ids"],
#                                     encoding["attention_mask"])
#     test_prediction = test_prediction.flatten().numpy()
#     print("Prediction for :", test_comment)
#     for label, prediction in zip(labels, test_prediction):
#         if prediction < threshold:
#             continue
#         print(f"{label}: {prediction}")

if __name__ == '__main__':
    dm = Pre_Normal_NotSeperated()

    labels_wps = [
        'HS', 
        'Abusive', 
        'HS_Individual', 
        'HS_Group', 
        'HS_Religion', 
        'HS_Race', 
        'HS_Physical', 
        'HS_Gender', 
        'HS_Other', 
        'HS_Weak', 
        'HS_Moderate', 
        'HS_Strong',
        'PS'
    ]
    model = XLM(labels_wps)
        #langsung twitter
        #coba pake automodel & autotokenizer
        #training gausah return metric


    early_stop_callback = EarlyStopping(monitor = 'val_loss', 
                                        min_delta = 0.001,
                                        patience = 3,
                                        mode = "min")
    
    logger = TensorBoardLogger("logs", name="bert_nalar")
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/bert', monitor='val_loss', mode='min')


    trainer = pl.Trainer(accelerator="gpu",
                        devices=1,
                        max_epochs = 15,
                        logger = logger,
                        log_every_n_steps=1,
                        default_root_dir = "./checkpoints/labels",
                        callbacks = [early_stop_callback, checkpoint_callback])
    
    print("================TRAINING================")

    trainer.fit(model, datamodule = dm)
    trainer.test(model = model, datamodule = dm)
    

    # print("Predictor")
    # predictor(trainer, labels)
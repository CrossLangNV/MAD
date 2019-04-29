#!/usr/bin/env python3

import os
import sys
import pandas as pd
import base64
import os
import string
import csv
import argparse

import os
import pandas as pd
import numpy as np
import torch
import pickle
import sys

from Levenshtein import ratio
import Levenshtein
import langdetect
import re

from scipy.stats import pearsonr
from sklearn.metrics import precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, make_scorer, roc_auc_score 
from sklearn.metrics.pairwise import linear_kernel

from number_match import number_match_pair

from nltk.corpus import stopwords


sys.path.append( '/MAD/InferSent' )

                    
#Load in the InferSent model.
#note: you need to use pytorch 0.4 to work with InferSent.
from models import InferSent


#Some functions to calculate the features:
def levenshtein(pair_of_sentences): 
    l = ratio(pair_of_sentences[0].lower(), pair_of_sentences[1].lower())
    assert l >= 0 and l <=1
    return l

def number_match(pair_of_sentences): return number_match_pair(pair_of_sentences)

def senlength(pair_of_sentences): return [len(pair_of_sentences[0]), len(pair_of_sentences[1])]


def calc_semantic_sim_infersent_batches( src_2_en, tgt_2_en, bs=1000, mini_batch_size=50 ):
    '''
    bs is the batch size so the calculation of linear kernel does not go out of memory. mini_batch_size is the batch size of InferSent. 
    '''    
    all_semantic_similarity = np.empty((  0  ))

    for i in range(0, len(src_2_en) , bs):
        print('sentences processed', i )

        embedding_src_2_en=model_infersent.encode(   src_2_en[i:i+bs]  , bsize=mini_batch_size, tokenize=True, verbose=False)
        embedding_tgt_2_en=model_infersent.encode(   tgt_2_en[i:i+bs]  , bsize=mini_batch_size, tokenize=True, verbose=False)

        semantic_similarity=np.diag( linear_kernel( embedding_src_2_en, embedding_tgt_2_en )) / (np.linalg.norm(  embedding_src_2_en, axis=1 ) * np.linalg.norm( embedding_tgt_2_en, axis=1    )  )

        all_semantic_similarity=np.append(  all_semantic_similarity,  semantic_similarity  )
    return all_semantic_similarity



def loss_r2(gold_lbls, pred_lbls): return (r2_score(gold_lbls,pred_lbls))  #AD: only use r2
scorer = make_scorer(loss_r2, greater_is_better=True)

#some helper functions:
def utility(gold_cls, pred_cls, p=0.33): return ( true_negative_rate(gold_cls, pred_cls)**(1-p)*recall_score(gold_cls, pred_cls)**p )  #AD: Calculate mean biased towards recall score.
def true_negative_rate(gold_cls, pred_cls): return confusion_matrix(gold_cls, pred_cls)[0,0]/(confusion_matrix(gold_cls, pred_cls)[0,0]+confusion_matrix(gold_cls, pred_cls)[0,1])
def root_relative_square_error(gold, pred): 
    from sklearn.metrics import mean_squared_error
    return (    mean_squared_error( gold  , pred  ) / mean_squared_error( gold  , np.repeat(np.mean(gold), len(gold)  )  )    )**0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_src", dest="path_src",
                        help="path to src sentences", required=True)
    parser.add_argument("--path_tgt", dest="path_tgt",
                    help="path to tgt sentences", required=True)
    parser.add_argument("--path_tgt_translated", dest="path_tgt_translated",
                help="path to tgt sentences translated in src languages", required=True)
    parser.add_argument("--output_folder", dest="output_folder",
            help="path to output_folder", required=True)
    parser.add_argument("--path_to_classifier", dest="path_to_classifier",
        help="path to classifier", required=False , default='/MAD/models/estimator_SVC.save' )
    parser.add_argument("--threshold_MAD", dest="threshold_MAD",
    help="threshold for MAD", required=False , type=float, default= 2.21  )
    
    args = parser.parse_args()
    output_folder=args.output_folder
    chosen_threshold=args.threshold_MAD #this was the optimal threshold found in 1-MAD_notebook.ipynb. This should actually be tuned on BLEU.
    
    #Load InferSent model
    model_version = 2
    MODEL_PATH = "/MAD/InferSent/encoder/infersent%s.pkl" % model_version  #AD: load in the InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model_infersent = InferSent(params_model)
    model_infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH='/MAD/InferSent/dataset/crawl-300d-2M.vec'
    model_infersent.set_w2v_path(W2V_PATH)

    # Load embeddings of K most frequent words
    model_infersent.build_vocab_k_words(K=1000000)

    print("InferSent model loaded")
    
    #input: src, tgt, tgt.translated (to src, being English).


    src=open( args.path_src , "r"  ).read().split("\n")
    src=src[:-1]
    tgt=open( args.path_tgt , "r"  ).read().split("\n")
    tgt=tgt[:-1]
    
    Txt_target_2_cross=open(  args.path_tgt_translated  ).read().split("\n")
    Txt_target_2_cross=Txt_target_2_cross[:-1]

    assert len( src ) == len( tgt )
    assert len( src ) == len( Txt_target_2_cross )

    #make a pandas dataframe of the data

    data = pd.DataFrame(
        {'Txt_source_0ri': src,
         'Txt_target_0ri': tgt,
         'Txt_target_2_cross': Txt_target_2_cross
        })

    data['Ft_source_cross_levenshtein'] = data.apply(lambda row: levenshtein((row['Txt_target_2_cross'], row['Txt_source_0ri'])) , axis=1)

    #data['Ft_en_levenshtein'] = data.apply(lambda row: levenshtein((row['Txt_source_2_en_sys'], row['Txt_target_2_en_sys'])) , axis=1)

    data['Ft_number_match'] = data.apply(lambda row: number_match(  (row['Txt_source_0ri'], row['Txt_target_0ri'])) , axis=1)

    data['Ft_senlength_av'] = data.apply(lambda row: np.mean( senlength( ( row[ 'Txt_source_0ri'] , row[ 'Txt_target_0ri'  ] )  ) )  , axis=1)

    data[ 'Ft_senlength_diff' ]=data.apply(lambda row: abs(  senlength( ( row[ 'Txt_source_0ri' ], row[ 'Txt_target_0ri'  ] )  ) [0] -  \
                                                                senlength( ( row[ 'Txt_source_0ri' ], row[ 'Txt_target_0ri'  ] )  )[1]   ), axis=1)

    #data['Ft_semantic_sim_bag_of_words'] = data.apply(lambda row: calc_cos_sim_bag_of_words((row['Txt_source_2_en_sys'], row['Txt_target_2_en_sys']) ,wv_model_en  ) , axis=1)

    print("calculate semantic similarity")

    data[ 'Ft_semantic_sim_infersent' ]  = calc_semantic_sim_infersent_batches( data['Txt_source_0ri'].tolist() , data['Txt_target_2_cross'].tolist()  )

    X=data.filter(regex='^Ft_.*').values  #Get all the features
    feature_names=data.filter(regex='^Ft_.*').columns

    #the chosen features should match the features the estimator was trained on. 
    chosen_features=[ 'Ft_source_cross_levenshtein'  , 'Ft_number_match' , 'Ft_senlength_av',  'Ft_senlength_diff', 'Ft_semantic_sim_infersent'   ]

    chosen_features_indices = []
    for i in range( len( feature_names ) ):
        if feature_names[i] in chosen_features:
            chosen_features_indices.append( i )

    #Load the trained estimator.

    estimator = pickle.load(open(  args.path_to_classifier , 'rb'))

    predicted_labels=estimator.predict(X[:, chosen_features_indices ]  )
    predicted_classification=(predicted_labels> chosen_threshold ).astype(int)  #AD: so if predicted label is larger than t==>then classified as misaligned. 

    data['predicted_classification']=predicted_classification
    data['predicted_labels']=predicted_labels

    print(  'classified as aligned %s or %.4f'   %( ( len(predicted_classification) -np.sum(predicted_classification) )  \
                                                     , ( len(predicted_classification) -np.sum(predicted_classification) ) / len(predicted_classification)     )   )
    print(  'classified as misaligned  %s or %.4f '  %(  np.sum(predicted_classification)   , ( np.sum(predicted_classification) ) / len(predicted_classification)     ) )


    MAD_scores=data.predicted_labels.tolist() 

    if os.path.exists( os.path.join(  output_folder , f'MAD_scores'  ) ):
        os.remove( os.path.join(  output_folder , f'MAD_scores' )  ) 

    f = open( os.path.join( output_folder , f'MAD_scores'    ), "a")


    for MAD_score in MAD_scores:
        f.write( "{0}\n".format( MAD_score  ) )    
    f.close()

    

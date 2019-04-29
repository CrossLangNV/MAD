# MAD
Misalignment detection (MAD)

Repository contains code for the paper: Misalignment detection for web scraped corpora: a supervised regression approach.

### Dependencies
python3 <br/>
numpy <br/>
scipy <br/>
scikit-learn <br/>
pandas <br/>
pytorch (tested with 0.4) <br/>
InferSent (git clone https://github.com/facebookresearch/InferSent.git ) <br/>

Dockerfile can be found in docker_mad/Dockerfile <br/>

### Labeled dataset
Labeled dataset used to train supervised regression model:   <br/>
DATA/supervised_dataset/labeled.tsv  <br/>

A tab separated file, with at each line:  <br/>
source /t target /t source_2_cross /t target_2_cross /t source_2_en /t target_2_en /t MAD_score  <br/>

### Notebook

Notebook where machine learning pipeline is explained:  <br/>
https://github.com/ArneDefauw/MAD/blob/master/1-MAD_notebook.ipynb

### Pre-trained MAD model
MAD/models/estimator_SVC.save

### Using MAD
MAD.py

example:

if en.sentences is a file containing English sentences; ga.sentences is file containing Irish sentences; and ga.sentences.translated contains translation of ga.sentences to English, then we can calculate MAD score for each sentence pair via the command:


python MAD.py \

--path_src en.sentences   \

--path_tgt ga.sentences \

--path_tgt_translated ga.sentence.translated  \

--output_folder /MAD


Code will create the file MAD_scores with at each line the MAD score for the corresponding sentence pair.

### Data used for intrinsic evaluation

The folder 'MAD/Gold_standard' contains the data used for intrinsic evaluation on alignment Gold Standard (EN-FR and EN-GA).  <br/>

The folders 'documents_en_fr' and 'documents_en_ga' contain the 13 respectively 11 documents used for creation of Gold Standards. The file 'url_keys_matched' is a tab separated file containing the doc id, the url of the English documents, url of the foreign language documents and the similarity score produced by Malign (https://github.com/paracrawl/Malign)

'corpus_en_fr' and 'corpus_en_ga' are tab separated files containing the alignments produced by Hunalign with at each line:

url_src /t url_tgt /t src /t tgt /t BiCleaner_score /t MAD_score


### Data used for extrinsic evaluation

Data used for training of our NMT engines can be found here:
http://www.crosslang.com/sites/crosslang/files/corpus_en_fr.gz
http://www.crosslang.com/sites/crosslang/files/corpus_en_ga.gz

They are tab separated files with at each line

url_src /t url_tgt /t src /t tgt /t BiCleaner_score /t MAD_score




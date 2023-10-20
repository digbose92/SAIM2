import os 
import pandas as pd 
import numpy as np 
from sentence_transformers import SentenceTransformer
import pickle
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


filename="/data/Multi_Domain_Data/parsed_csv_data/dvd/negative_review.csv"
sbert_folder="/data/Multi_Domain_Data/SBERT_features/dvd"
csv_data=pd.read_csv(filename)
sentence_list=[]
object_id_list=[]
review_list=[]
for i in np.arange(csv_data.shape[0]):  
    sentence_list.append(csv_data['Text'].iloc[i])
    object_id_list.append(csv_data['Object_ID'].iloc[i])
    review_list.append(csv_data['Rating'].iloc[i])

embeddings = model.encode(sentence_list)
embedding_list=[]

for sentence, embedding in zip(sentence_list, embeddings):
    index_sentence=sentence_list.index(sentence)
    object_id_name=object_id_list[index_sentence]
    embedding_list.append(embedding)


embedding_array=np.array(embedding_list)
embedding_dict={'Embedding':embedding_array,'Ratings':review_list}
with open(os.path.join(sbert_folder,filename.split("/")[-1].split(".")[0]+".pkl"),"wb") as f:
    pickle.dump(embedding_dict,f)
    #np.save(os.path.join(sbert_folder,object_id_name+".npy"),embedding)
    #with open(os.path.join(sbert_folder,object_id_name+".npy"),"wb") as f:

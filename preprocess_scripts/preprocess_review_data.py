import os 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
from collections import Counter
import json 
import codecs
# def extract_fields_data(sub_list):
#     #print(sub_list)
#     print(sub_list)
#     unique_id_list_start=sub_list.append('<unique_id>\n')
#     unique_id_list_end=sub_list.append('</unique_id>\n')
    
#     object_id=sub_list[2]
#     asin=sub_list[5]
#     product_name=sub_list[8]
#     product_type=sub_list[11]
#     rating=sub_list[17]
#     title=sub_list[20]
#     date=sub_list[23]
#     reviewer=sub_list[26]
#     reviewer_location=sub_list[29]
#     review_text=sub_list[32]

#     sample_dict={'Object_ID':[object_id],
#     'Asin':[asin],
#     'Product_name':[product_name],
#     'Product_type':[product_type],
#     'Rating':[rating],'Title':[title],
#     'Date':[date],
#     'Reviewer':[reviewer],
#     'Reviewer_location':[reviewer_location],
#     'Text':[review_text]}

#     #print(sample_dict)
#     return(sample_dict)
    #print(product_name)

def extract_fields_data(sub_list):

    
    unique_id_start_indices=[i for i, x in enumerate(sub_list) if x == '<unique_id>\n']
    unique_id_end_indices=[i for i, x in enumerate(sub_list) if x == '</unique_id>\n']

    asin_start_indices=[i for i, x in enumerate(sub_list) if x == '<asin>\n']
    asin_end_indices=[i for i, x in enumerate(sub_list) if x == '</asin>\n']

    product_name_start_indices=[i for i, x in enumerate(sub_list) if x == '<product_name>\n']
    product_name_end_indices=[i for i, x in enumerate(sub_list) if x == '</product_name>\n']

    product_type_start_indices=[i for i, x in enumerate(sub_list) if x == '<product_type>\n']
    product_type_end_indices=[i for i, x in enumerate(sub_list) if x == '</product_type>\n']

    rating_start_indices=[i for i, x in enumerate(sub_list) if x == '<rating>\n']
    rating_end_indices=[i for i, x in enumerate(sub_list) if x == '</rating>\n']

    title_start_indices=[i for i, x in enumerate(sub_list) if x == '<title>\n']
    title_end_indices=[i for i, x in enumerate(sub_list) if x == '</title>\n']

    date_start_indices=[i for i, x in enumerate(sub_list) if x == '<date>\n']
    date_end_indices=[i for i, x in enumerate(sub_list) if x == '</date>\n']

    reviewer_start_indices=[i for i, x in enumerate(sub_list) if x == '<reviewer>\n']
    reviewer_end_indices=[i for i, x in enumerate(sub_list) if x == '</reviewer>\n']

    reviewer_location_start_indices=[i for i, x in enumerate(sub_list) if x == '<reviewer_location>\n']
    reviewer_location_end_indices=[i for i, x in enumerate(sub_list) if x == '</reviewer_location>\n']

    review_text_start_indices=[i for i, x in enumerate(sub_list) if x == '<review_text>\n']
    review_text_end_indices=[i for i, x in enumerate(sub_list) if x == '</review_text>\n']

    #sample list based on this data
    unique_id_list=sub_list[unique_id_start_indices[0]+1:unique_id_end_indices[0]]
    asin_list=sub_list[asin_start_indices[0]+1:asin_end_indices[0]]
    product_name_list=sub_list[product_name_start_indices[0]+1:product_name_end_indices[0]]
    product_type_list=sub_list[product_type_start_indices[0]+1:product_type_end_indices[0]]
    rating_list=sub_list[rating_start_indices[0]+1:rating_end_indices[0]]
    title_list=sub_list[title_start_indices[0]+1:title_end_indices[0]]
    date_list=sub_list[date_start_indices[0]+1:date_end_indices[0]]
    reviewer_list=sub_list[reviewer_start_indices[0]+1:reviewer_end_indices[0]]
    reviewer_location_list=sub_list[reviewer_location_start_indices[0]+1:reviewer_location_end_indices[0]]
    review_text_list=sub_list[review_text_start_indices[0]+1:review_text_end_indices[0]]


    #print them 
    # print(unique_id_list)
    # print(asin_list)
    # print(product_name_list)
    # print(product_type_list)
    # print(rating_list)
    # print(title_list)
    # print(date_list)
    # print(reviewer_list)
    # print(reviewer_location_list)
    # print(review_text_list)

    #convert to string
    subject_dict={'Object_ID':[u.split("\n")[0] for u in unique_id_list],
    'Asin':[u.split("\n")[0] for u in asin_list],
    'Product_name':[u.split("\n")[0] for u in product_name_list],
    'Product_type':[u.split("\n")[0] for u in product_type_list],
    'Rating':[u.split("\n")[0] for u in rating_list],'Title':[u.split("\n")[0] for u in title_list],
    'Date':[u.split("\n")[0] for u in date_list],
    'Reviewer':[u.split("\n")[0] for u in reviewer_list],
    'Reviewer_location':[u.split("\n")[0] for u in reviewer_location_list],
    'Text':[u.split("\n")[0] for u in review_text_list]}

    #print(subject_dict)

    return(subject_dict)


def extract_review_data(review_data,review_indices):

    dict_list=[]
    label_list=[]
    for i in tqdm(np.arange(len(review_indices))):
        if(i==len(review_indices)-1):
            data_curr_sample=review_data[review_indices[i]:]
        else:
            data_curr_sample=review_data[review_indices[i]:review_indices[i+1]-1]
        sample_dict=extract_fields_data(data_curr_sample)
        if(int(float(sample_dict['Rating'][0]))==4 or int(float(sample_dict['Rating'][0]))==5):
            label_list.append(1) #positive review
            sample_dict['Label']=[1]
        elif(int(float(sample_dict['Rating'][0]))<3):
            label_list.append(0) #negative review 
            sample_dict['Label']=[0]
        #print(sample_dict)
        dict_list.append(sample_dict)

    return(dict_list)
    

domain='electronics'
positive_review_data_path="/scratch1/dbose/Multi_Domain_data_gcp/redownloaded_data/sorted_data/"+domain+"/positive.review"
negative_review_data_path="/scratch1/dbose/Multi_Domain_data_gcp/redownloaded_data/sorted_data/"+domain+"/negative.review"
dest_filename=domain+"_review.json"
print(dest_filename)

with open(positive_review_data_path,'rb') as f:
    positive_review_data=f.readlines()

with open(negative_review_data_path,'rb') as f:
    negative_review_data=f.readlines()

#print(negative_review_data[0:10])
pos_review_data=[str(r,'latin-1') for r in positive_review_data]
print(pos_review_data[0:10])
negative_review_data=[str(r,'latin-1') for r in negative_review_data]
print(negative_review_data[0:10])
# print(len(postive_review_data),len(negative_review_data))

positive_review_indices=[i for i,v in enumerate(pos_review_data) if v == '<review>\n']
negative_review_indices=[i for i,v in enumerate(negative_review_data) if v == '<review>\n']
# print(len(positive_review_indices),len(negative_review_indices))

positive_dict=extract_review_data(pos_review_data,positive_review_indices)

#print(len(positive_review_indices),len(positive_review_data))
negative_dict=extract_review_data(negative_review_data,negative_review_indices)
print(len(negative_dict),len(positive_dict))

# print(len(negative_dict),len(positive_dict))
total_dict=positive_dict+negative_dict
#dump the total dict into a json file 
dest_folder="/scratch1/dbose/Multi_Domain_data_gcp/redownloaded_data/parsed_data"
with open(os.path.join(dest_folder,dest_filename),'w') as f:
    json.dump(total_dict,f)

# print(len(total_dict))

#print(review_indices)
#extract everything between <review> and </review>

#fields to be used each column in df - unique_id, asin, product name, product type, helpful, rating, title, date , reviewer, reviewer location, review_text 
#print(review_data[review_indices[0]:review_indices[1]-2])
# label_list=[]
# df_data_frame=[]
# for i in tqdm(np.arange(len(review_indices))):
    
#     if(i==len(review_indices)-1):
#         data_curr_sample=review_data[review_indices[i]:]
#     else:
#         data_curr_sample=review_data[review_indices[i]:review_indices[i+1]-1]
#     sample_dict=extract_fields_data(data_curr_sample)
#     if(int(float(sample_dict['Rating'][0]))==4 or int(float(sample_dict['Rating'][0]))==5):
#         label_list.append(1) #positive review
#     elif(int(float(sample_dict['Rating'][0]))<3):
#         label_list.append(0) #negative review 


# print(Counter(label_list))
# data_curr_sample=review_data[review_indices[-1]:]
# print(data_curr_sample)

# data_ext_sample=review_data[review_indices[-1]:]
# sample_fin_dict=extract_fields_data(data_ext_sample)
# sample_fin_df=pd.DataFrame.from_dict(sample_fin_dict)
# df_data_frame.append(sample_fin_df)
# data_df=pd.concat(df_data_frame)

# data_df.to_csv(os.path.join(dest_subfolder,dest_filename),index=False)




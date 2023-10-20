#electronics model feature space is dvd

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_electronics/20211222-024850_bert-base-uncased/20211222-024850_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_electronics/20211222-024850_bert-base-uncased/20211222-024850_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/electronics --csv_file /data/Multi_Domain_Data/parsed_csv_data/dvd/dvd_review_splits_combined.csv

#electronics model feature space is books

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_electronics/20211222-024850_bert-base-uncased/20211222-024850_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_electronics/20211222-024850_bert-base-uncased/20211222-024850_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/electronics --csv_file /data/Multi_Domain_Data/parsed_csv_data/books/books_review_splits_combined.csv

#electronics model feature space is electronics

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_electronics/20211222-024850_bert-base-uncased/20211222-024850_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_electronics/20211222-024850_bert-base-uncased/20211222-024850_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/electronics --csv_file /data/Multi_Domain_Data/parsed_csv_data/electronics/electronics_review_splits_combined.csv


#electronics model feature space is kitchen

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_electronics/20211222-024850_bert-base-uncased/20211222-024850_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_electronics/20211222-024850_bert-base-uncased/20211222-024850_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/electronics --csv_file /data/Multi_Domain_Data/parsed_csv_data/kitchen_\&_housewares/kitchen_\&_housewares_review_splits_combined.csv

#books model feature space is dvd

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_books/20211222-030738_bert-base-uncased/20211222-030738_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_books/20211222-030738_bert-base-uncased/20211222-030738_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/books --csv_file /data/Multi_Domain_Data/parsed_csv_data/dvd/dvd_review_splits_combined.csv

#books model feature space is electronics

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_books/20211222-030738_bert-base-uncased/20211222-030738_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_books/20211222-030738_bert-base-uncased/20211222-030738_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/books --csv_file /data/Multi_Domain_Data/parsed_csv_data/electronics/electronics_review_splits_combined.csv

#books model feature space is books

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_books/20211222-030738_bert-base-uncased/20211222-030738_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_books/20211222-030738_bert-base-uncased/20211222-030738_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/books --csv_file /data/Multi_Domain_Data/parsed_csv_data/books/books_review_splits_combined.csv

#books model feature space is kichen 

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_books/20211222-030738_bert-base-uncased/20211222-030738_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_books/20211222-030738_bert-base-uncased/20211222-030738_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/books --csv_file /data/Multi_Domain_Data/parsed_csv_data/kitchen_\&_housewares/kitchen_\&_housewares_review_splits_combined.csv

#kitchen model feature space is books

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_kitchen_\&_housewares/20211222-035307_bert-base-uncased/20211222-035307_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_kitchen_\&_housewares/20211222-035307_bert-base-uncased/20211222-035307_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/kitchen_\&_housewares --csv_file /data/Multi_Domain_Data/parsed_csv_data/books/books_review_splits_combined.csv

#kitchen model feature space is electronics

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_kitchen_\&_housewares/20211222-035307_bert-base-uncased/20211222-035307_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_kitchen_\&_housewares/20211222-035307_bert-base-uncased/20211222-035307_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/kitchen_\&_housewares --csv_file /data/Multi_Domain_Data/parsed_csv_data/electronics/electronics_review_splits_combined.csv

#kitchen model feature space is dvd

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_kitchen_\&_housewares/20211222-035307_bert-base-uncased/20211222-035307_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_kitchen_\&_housewares/20211222-035307_bert-base-uncased/20211222-035307_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/kitchen_\&_housewares --csv_file /data/Multi_Domain_Data/parsed_csv_data/dvd/dvd_review_splits_combined.csv

#kitchen model feature space is kitchen

CUDA_VISIBLE_DEVICES=0,1,2,3 python extract_BERT_finetuned_model_features.py --model_option /home/dbose_usc_edu/data/domain_adapt_data/model_dir/bert-base-uncased_kitchen_\&_housewares/20211222-035307_bert-base-uncased/20211222-035307_bert-base-uncased_best_model.pt --config_file /home/dbose_usc_edu/data/domain_adapt_data/log_dir/bert-base-uncased_kitchen_\&_housewares/20211222-035307_bert-base-uncased/20211222-035307_bert-base-uncased.yaml --dest_folder /data/Multi_Domain_Data/BERT_features/kitchen_\&_housewares --csv_file /data/Multi_Domain_Data/parsed_csv_data/kitchen_\&_housewares/kitchen_\&_housewares_review_splits_combined.csv
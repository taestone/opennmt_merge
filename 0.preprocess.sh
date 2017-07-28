data_home=~/data/crossref
#th preprocess.lua -train_src ${data_home}/crossref_text_1000.txt -train_tgt ${data_home}/crossref_title_1000.txt -valid_src ${data_home}/crossref_text_1000.txt -valid_tgt ${data_home}/crossref_text_1000.txt -save_data ./mix_1000/mix
#th preprocess.lua -train_src ${data_home}/crossref_text_10000.txt -train_tgt ${data_home}/crossref_title_10000.txt -valid_src ${data_home}/crossref_text_1000.txt -valid_tgt ${data_home}/crossref_text_1000.txt -save_data ./mix_10000/mix
#th preprocess.lua -idx_files true -train_src ${data_home}/crossref_idx_text_100000.txt -train_tgt ${data_home}/crossref_idx_title_100000.txt -valid_src ${data_home}/crossref_idx_text_10000.txt -valid_tgt ${data_home}/crossref_idx_text_10000.txt -save_data ./output/mix
th preprocess.lua -idx_files true -train_src ${data_home}/crossref_idx_text_100000.txt -train_tgt ${data_home}/crossref_idx_title_100000.txt -valid_src ${data_home}/crossref_idx_text_10000.txt -valid_tgt ${data_home}/crossref_idx_text_10000.txt -save_data ./mix_100000/mix -tgt_vocab output/tgt.old.dict -src_vocab output/src.old.dict -src_vocab_size 49999 -tgt_vocab_size 49999

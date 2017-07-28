model=mix_100000
data=mix_100000
orig_model=models/textsum_epoch7_14.69_release.t7
#orig_model=./mix_100000_new/mix_100000_new_epoch1_1399.97.t7
#th train.lua -data $model/mix-train.t7 -save_model $data/$model -gpuid 1
th train.lua -train_from $orig_model -data $model/mix-train.t7 -save_model $data/$model  -end_epoch 10000 -fp16 true -continue -rnn_size 300 -log_level DEBUG -word_vec_size 500

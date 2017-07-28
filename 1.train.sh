model=mix_100000
data=mix_100000
orig_model=./models/textsum_epoch7_14.69_release.t7
#orig_model=~/models/opennmt/sum_model_epoch11_14.62.t7
#th train.lua -data $model/mix-train.t7 -sae_model $data/$model -gpuid 1
th train.lua -train_from $orig_model -data $model/mix-train.t7 -save_model $data/$model -end_epoch 10000 -continue -rnn_size 300 -log_level DEBUG -word_vec_size 500
#th train.lua -train_from $orig_model -data $model/mix-train.t7 -save_model $data/$model -end_epoch 10000 -continue -rnn_size 300 -log_level DEBUG -word_vec_size 500
#th train.lua -data $model/mix-train.t7 -train_from $orig_model -save_model $data/$model -gpuid 0


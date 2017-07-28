model=mix_100000
data=mix_100000
orig_model=./textsum_epoch13_906.32.t7
#orig_model=~/models/opennmt/sum_model_epoch11_14.62.t7
#th train.lua -data $model/mix-train.t7 -sae_model $data/$model -gpuid 1
th train.lua -data $model/mix-train.t7 -train_from $orig_model -save_model $data/$model -gpuid 0

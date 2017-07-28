cat bpe_title.out > bpe_merge_title.txt
head -n $1 ~/data/sumdata/train/train.title.txt >> bpe_merge_title.txt
cat bpe.out > bpe_merge_text.txt
head -n $1 ~/data/sumdata/train/train.article.txt >> bpe_merge_text.txt
sh 0.preprocess.sh 

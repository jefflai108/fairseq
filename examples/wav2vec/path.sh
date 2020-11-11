source /data/sls/scratch/clai24/anaconda3/etc/profile.d/conda.sh
conda activate libri-light
#export PYTHONPATH="${PYTHONPATH}:/data/sls/temp/clai24/contrastive-learning/pytorch-lightning-bolts"

# Kaldi related 
ESPNET_ROOT=/data/sls/temp/clai24/amazon_intern/espnet
KALDI_ROOT=$ESPNET_ROOT/tools/kaldi
export PATH=/data/sls/scratch/swshon/tools/kaldi-trunk/kaldi_july2018/egs/wsj/s5/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh


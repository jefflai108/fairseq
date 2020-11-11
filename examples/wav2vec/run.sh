#!/bin/bash 

stage=$1
ngpu=1

if [ $stage -eq 0 ]; then 
    for split in dev-clean dev-other train-clean-100 test-clean test-other train-clean-360 train-960; do 
        mkdir data/${split}
        python wav2vec_manifest.py /data/sls/temp/clai24/data/LibriSpeech/${split} --dest data/${split} --ext flac --valid-percent 0
    done
fi 

if [ $stage -eq 1 ]; then   
    for split in dev-clean dev-other train-clean-100 test-clean test-other train-clean-360 train-960; do 
        python libri_labels.py data/${split}/train.tsv --output-dir data/${split} --output-name $split
    done
fi 

if [ $stage -eq 2 ]; then 
	## fine-tuning commands
    # hyper-parameter for fine-tuning on 100 hr (Table 6 in https://arxiv.org/pdf/2006.11477.pdf)
    # validate on the dev-clean subset
    train_subset=train-clean-100
    valid_subset=dev-clean
    expdir=exp/train-clean-100-debug
    datadir=data/debug/
    python /data/sls/temp/clai24/knowledge-transfer/fairseq/train.py --distributed-world-size ${ngpu} --distributed-port 0 $datadir --save-dir $expdir \
        --post-process letter --train-subset $train_subset --valid-subset $valid_subset --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 8 \
        --max-update 50000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path /data/sls/temp/clai24/pretrained-models/wav2vec_small.pt \
        --labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
        --mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.512 --zero-infinity \
        --feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
        --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 5000 --hold-steps 20000 \
        --decay-steps 25000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
        --attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --log-format tqdm --log-interval=100 --ddp-backend no_c10d \
        --update-freq $(( 24/$ngpu )) 2>&1 | tee $expdir/train.log
fi 


if [ $stage -eq 3 ]; then
	## train from scratch commands
    train_subset=train-960
    valid_subset=dev-clean
    expdir=exp/${train_subset}/att_head_6-embed_dim_480-ffn_embed_dim_1920-layer_6-latent_vars_160
    datadir=data/${train_subset}
    mkdir -p $expdir

	python /data/sls/temp/clai24/knowledge-transfer/fairseq/train.py --distributed-world-size ${ngpu} --distributed-port 0 $datadir --save-dir $expdir \
        --train-subset $train_subset --valid-subset $valid_subset \
		--num-workers 8 --task audio_pretraining --criterion wav2vec_kd --arch wav2vec2 \
		--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
		--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
		--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
		--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
		--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
		--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
		--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
		--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
		--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
        --encoder-attention-heads 6 --encoder-embed-dim 480 --encoder-ffn-embed-dim 1920 --encoder-layers 6 --latent-vars 160 2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 4 ]; then
	## train from scratch commands
    train_subset=train-960
    valid_subset=dev-clean
    expdir=exp/${train_subset}/att_head_6-embed_dim_480-ffn_embed_dim_1920-layer_3-latent_vars_160
    datadir=data/${train_subset}
    mkdir -p $expdir

	python /data/sls/temp/clai24/knowledge-transfer/fairseq/train.py --distributed-world-size ${ngpu} --distributed-port 0 $datadir --save-dir $expdir \
        --train-subset $train_subset --valid-subset $valid_subset \
		--num-workers 8 --task audio_pretraining --criterion wav2vec_kd --arch wav2vec2 \
		--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
		--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
		--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
		--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
		--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
		--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
		--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
		--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
		--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
        --encoder-attention-heads 6 --encoder-embed-dim 480 --encoder-ffn-embed-dim 1920 --encoder-layers 3 --latent-vars 160 2>&1 | tee $expdir/train.log
fi


if [ $stage -eq 5 ]; then
	## train from scratch commands
    train_subset=train-960
    valid_subset=dev-clean
    expdir=exp/${train_subset}/load_extractor-att_head_6-embed_dim_480-ffn_embed_dim_1920-layer_6-latent_vars_160
    datadir=data/${train_subset}
    mkdir -p $expdir

	python /data/sls/temp/clai24/knowledge-transfer/fairseq/train.py --distributed-world-size ${ngpu} --distributed-port 0 $datadir --save-dir $expdir \
        --train-subset $train_subset --valid-subset $valid_subset \
		--num-workers 8 --task audio_pretraining --criterion wav2vec_kd --arch wav2vec2 \
		--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
		--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
		--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
		--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
		--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
		--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
		--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
		--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
		--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
        --load-extractor --encoder-attention-heads 6 --encoder-embed-dim 480 --encoder-ffn-embed-dim 1920 --encoder-layers 6 --latent-vars 160 2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 6 ]; then
	## train from scratch commands
    train_subset=train-960
    valid_subset=dev-clean
    expdir=exp/${train_subset}/load_extractor-att_head_6-embed_dim_480-ffn_embed_dim_1920-layer_3-latent_vars_160
    datadir=data/${train_subset}
    mkdir -p $expdir

	python /data/sls/temp/clai24/knowledge-transfer/fairseq/train.py --distributed-world-size ${ngpu} --distributed-port 0 $datadir --save-dir $expdir \
        --train-subset $train_subset --valid-subset $valid_subset \
		--num-workers 8 --task audio_pretraining --criterion wav2vec_kd --arch wav2vec2 \
		--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
		--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
		--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
		--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
		--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
		--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
		--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
		--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
		--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
        --load-extractor --encoder-attention-heads 6 --encoder-embed-dim 480 --encoder-ffn-embed-dim 1920 --encoder-layers 3 --latent-vars 160 2>&1 | tee $expdir/train.log
fi

# original 
if [ $stage -eq -100 ]; then
	## train from scratch commands
    train_subset=train-clean-100
    valid_subset=dev-clean
    expdir=exp/${train_subset}/debug
    datadir=data/${train_subset}
    mkdir -p $expdir

	python /data/sls/temp/clai24/knowledge-transfer/fairseq/train.py --distributed-world-size ${ngpu} --distributed-port 0 $datadir --save-dir $expdir \
        --train-subset $train_subset --valid-subset $valid_subset \
		--num-workers 8 --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
		--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
		--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
		--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
		--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
		--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
		--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
		--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
		--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
		--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d 2>&1 | tee $expdir/train.log
fi


# dummy run 
if [ $stage -eq 100 ]; then
	## train from scratch commands
    train_subset=train-960
    valid_subset=dev-clean
    expdir=exp/debug
    datadir=data/debug
    mkdir -p $expdir

	python /data/sls/temp/clai24/knowledge-transfer/fairseq/train.py --distributed-world-size ${ngpu} --distributed-port 0 $datadir --save-dir $expdir \
        --train-subset $train_subset --valid-subset $valid_subset \
		--num-workers 8 --task audio_pretraining --criterion wav2vec_kd --arch wav2vec2 \
		--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
		--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
		--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
		--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
		--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
		--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
		--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
		--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
		--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
        --encoder-attention-heads 6 --encoder-embed-dim 480 --encoder-ffn-embed-dim 1920 --encoder-layers 6 --latent-vars 160 2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 100 ]; then 
    WAV_PATH=test.wav
    TARGET_DICT_PATH=/data/sls/temp/clai24/amazon_intern/fairseq/examples/wav2vec/data/train-clean-100/dict.ltr.txt
    #python recognize.py --wav_path $WAV_PATH --w2v_path exp/train-clean-100/checkpoint_best.pt --target_dict_path $TARGET_DICT_PATH

	python /data/sls/temp/clai24/amazon_intern/fairseq/examples/speech_recognition/infer.py data/train-clean-100/ --task audio_pretraining \
		--nbest 1 --path exp/train-clean-100/checkpoint_best.pt --results-path exp/train-clean-100/decode --w2l-decoder viterbi \
		--word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
		--post-process letter
fi 

#!/bin/bash

python main.py \
    --debug False \
    --resume_path "../saved/mimic3/trained_model_0.4799" \
    --Test False \
    --dataset "mimic3" \
    --device 0 \
    --dp 0.5 \
    --lr 0.0005 \
    --dim 64 \
    --regular 0.05 \
    --epochs 20 \
    --seed 1 \
    --savepath "data/saved_demo_seed1/" \
    --usedemographics True \
    --usenotes False \
    --uselabevents False

python main.py \
    --debug False \
    --resume_path "../saved/mimic3/trained_model_0.4799" \
    --Test False \
    --dataset "mimic3" \
    --device 0 \
    --dp 0.5 \
    --lr 0.0005 \
    --dim 64 \
    --regular 0.05 \
    --epochs 20 \
    --seed 18 \
    --savepath "data/saved_demo_seed18/" \
    --usedemographics True \
    --usenotes False \
    --uselabevents False

python main.py \
    --debug False \
    --resume_path "../saved/mimic3/trained_model_0.4799" \
    --Test False \
    --dataset "mimic3" \
    --device 0 \
    --dp 0.5 \
    --lr 0.0005 \
    --dim 64 \
    --regular 0.05 \
    --epochs 20 \
    --seed 3 \
    --savepath "data/saved_demo_seed3/" \
    --usedemographics True \
    --usenotes False \
    --uselabevents False

python main.py \
    --debug False \
    --resume_path "../saved/mimic3/trained_model_0.4799" \
    --Test False \
    --dataset "mimic3" \
    --device 0 \
    --dp 0.5 \
    --lr 0.0005 \
    --dim 64 \
    --regular 0.05 \
    --epochs 20 \
    --seed 16 \
    --savepath "data/saved_demo_seed16/" \
    --usedemographics True \
    --usenotes False \
    --uselabevents False

python main.py \
    --debug False \
    --resume_path "../saved/mimic3/trained_model_0.4799" \
    --Test False \
    --dataset "mimic3" \
    --device 0 \
    --dp 0.5 \
    --lr 0.0005 \
    --dim 64 \
    --regular 0.05 \
    --epochs 20 \
    --seed 1234 \
    --savepath "data/saved_demo_seed1234/" \
    --usedemographics True \
    --usenotes False \
    --uselabevents False

## running for 5 seeds
# None
## none1 saved_attn_seed1
## none2 saved_attn_seed18
## none3 saved_attn_seed3
## none4 saved_attn_seed16

# Only demographics
## demo1 saved_demo_seed1
## demo2 saved_demo_seed18
## demo3 saved_demo_seed3
## demo4 saved_demo_seed16

# Only labevents
## labevents1 saved_labevents_seed1
## labevents2 saved_labevents_seed18
## labevents3 saved_labevents_seed3
## labevents4 saved_labevents_seed16

# Both demographics and labevents
## both1 saved_labevents_demo_seed1
## both2 saved_labevents_demo_seed18
## both3 saved_labevents_demo_seed3
## both4 saved_labevents_demo_seed16


# seed 1234
# notes = F, labevents = F, demographics = F -> attn1 saved_attn
# notes = F, labevents = F, demographics = T -> attn2 saved_demo
# notes = F, labevents = T, demographics = F -> attn3 saved_labevents
# notes = F, labevents = T, demographics = T -> attn4 saved_labevents_demo


# notes = T, labevents = F, demographics = F -> attn5 saved_notes
# notes = T, labevents = F, demographics = T -> attn6 saved_notes_demo
# notes = T, labevents = T, demographics = F -> attn7 saved_notes_labevents
# notes = T, labevents = T, demographics = T -> attn8 saved_notes_labevents_demo
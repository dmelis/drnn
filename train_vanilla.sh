#!/bin/bash

if [ "$1" == "synth" ]; then
th main.lua -dataset synth -seq_len 20 -encoder LSTM -decoder tree,GRU,DTP \
            -batch 20 -epochs 5 -prediction_folds 'dev,test' -clip 5 -reg 0.0001 \
            -learning_rate 0.05 -memdim 50 -prediction -prediction_folds 'dev,test' \
            -dev_loss_every 1 -dev_pred_every 1 -select_criterion 'prediction' \
            -cropv 20000 -noprompt -debug 0 -embedding rand -maxexamples 50 -shuffle
            #-append_results -noprompt -maxexamples 10 -embedding rand
elif [ "$1" == "synth_STP" ]; then
th main.lua -dataset synth -seq_len 20 -encoder LSTM -decoder tree,GRU,STP \
            -batch 20 -epochs 20 -prediction_folds 'dev,test' -clip 5 -reg 0.0001 \
            -learning_rate 0.05 -memdim 50 -prediction -prediction_folds 'dev,test' \
            -dev_loss_every 1 -dev_pred_every 1 -select_criterion 'prediction' \
            -cropv 20000 -noprompt -embedding rand -maxexamples 50 -debug 1
            #-append_results -noprompt -maxexamples 10 -embedding rand
elif [ "$1" == "IFTTT" ]; then
  th main.lua -dataset IFTTT -seq_len 15 -encoder LSTM -decoder tree,GRU,DTP \
              -batch 25 -epochs 5 -prediction_folds 'dev,test' -clip 5 -reg 0.0001 \
              -learning_rate 0.06 -memdim 50 -prediction -prediction_folds 'dev,test' \
              -dev_loss_every 1 -dev_pred_every 1 -select_criterion 'prediction' \
              -cropv 20000 -noprompt -maxexamples 1000
              #-append_results -noprompt -maxexamples 10 -embedding rand
elif [ "$1" == "MT" ]; then
  th main.lua -source fr -target en -encoder LSTM -decoder tree,GRU,DTP \
              -batch 10 -epochs 20 -prediction_folds 'dev,test' -clip 5 -reg 0.0001 \
              -learning_rate 0.05 -memdim 50 -prediction -prediction_folds 'dev,test' \
              -dev_loss_every 1 -dev_pred_every 1 -select_criterion 'prediction' \
              -cropv 50000 -maxexamples 1000 -noprompt -debug 0 -test_generation \
              -embedding rand -debug 2
              #-append_results -noprompt -embedding rand
elif [ "$1" == "BABYMT" ]; then
  th main.lua -source fr -target en -encoder LSTM -decoder tree,GRU,DTP \
              -batch 25 -epochs 20 -prediction_folds 'dev,test' -clip 5 -reg 0.0001 \
              -learning_rate 0.01 -memdim 150 -prediction -prediction_folds 'dev,test' \
              -dev_loss_every 1 -dev_pred_every 1 -select_criterion 'prediction' \
              -cropv 20000 -debug 0 -test_generation -lr -seq_len 20 \
              -nosave  -maxexamples 50 #-embedding rand
              #-embedding rand -maxexamples 1000
              #-append_results -noprompt
fi

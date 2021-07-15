# Quality Estimation with BiRNNs (Word and Sentence Level)

1. First ensure the settings in `deepquestpy/config/birnn.jsonnet` (for sentence level experiments) or 
   `deepquestpy/config/birnn_word.jsonnet` (for word level experiments) are right for the setup.
   
2. In `train.sh`, set the `config_filename` variable to either of the config files in (1) above, and then run it to train a model.
   
3. Run `evaluate.sh` to evaluate the trained model
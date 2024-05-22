N_FOLDS=10
for FOLD in $(seq 0 $((N_FOLDS-1))); do
    python generate_splits_nct.py --config_path conf/cohort_selection_ssl.json --fold=$FOLD --n_folds=$N_FOLDS --output_filename splits/ssl_fold${FOLD}:${N_FOLDS}.json
    python generate_splits_nct.py --config_path conf/cohort_selection_main.json --fold=$FOLD --n_folds=$N_FOLDS --output_filename splits/fold${FOLD}:${N_FOLDS}.json
done
"""Main execution file. (In this template it simply runs train.py in src.modeling)"""

from src.config.config import (PROCESSED_DATA_DIR, RAW_DATA_DIR, LOAD_CACHED_DATA,
                               PROCESSED_DATA_FILENAME, RAW_DATA_FILENAME, RESULTS_DIR, DATA_LIMIT,
                               OPTIMIZATION_ITERATIONS_LIMIT, SHAP_SAMPLE_SIZE, TARGET_NAME,
                               TRAIN_BASELINE, TRAIN_ONLY_ONE_NON_BASELINE, SHAP_ON_BOTH_TEST_AND_TRAIN, TEST_SET_SIZE,
                               DECISION_BOUNDARY, CV_SPLITS, CV_SCORING_DICT, NON_MODELING_FIELDS_TO_DROP,
                               SAVE_PRODUCTION_PIPELINE, PRODUCTION_MODEL_NAME)
from src.config.estimators import MODEL_TRAINING_LIST
from src.modeling.train import master_train


def main():
    master_train(
        raw_path=RAW_DATA_DIR,
        raw_data_filename=RAW_DATA_FILENAME,
        process_path=PROCESSED_DATA_DIR,
        processed_data_filename=PROCESSED_DATA_FILENAME,
        results_path=RESULTS_DIR,
        model_training_list=MODEL_TRAINING_LIST,
        target_name=TARGET_NAME,
        non_modeling_fields_to_drop=NON_MODELING_FIELDS_TO_DROP,
        cv_strategy=CV_SPLITS,
        cv_scoring_dict=CV_SCORING_DICT,
        test_set_size=TEST_SET_SIZE,
        decision_boundary=DECISION_BOUNDARY,
        data_limit=DATA_LIMIT,
        train_baseline=TRAIN_BASELINE,
        train_only_one_baseline=TRAIN_ONLY_ONE_NON_BASELINE,
        optimization_iterations_limit=OPTIMIZATION_ITERATIONS_LIMIT,
        shap_on_both_test_and_train=SHAP_ON_BOTH_TEST_AND_TRAIN,
        shap_sample_size=SHAP_SAMPLE_SIZE,
        save_production_pipeline=SAVE_PRODUCTION_PIPELINE,
        production_model_name=PRODUCTION_MODEL_NAME,
        load_cached=LOAD_CACHED_DATA
    )


if __name__ == '__main__':
    main()

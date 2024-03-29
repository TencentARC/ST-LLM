export PYTHONPATH="./:$PYTHONPATH"
python stllm/test/gpt_evaluation/evaluate_activitynet_qa.py \
    --pred_path test_output/qabench/stllm_instructblipbase_anetqa.json \
    --output_dir test_output/qabench/activityQA/stllm_instructblipbase \
    --output_json test_output/qabench/activityQA/stllm_instructblipbase/activityQA.json \
    --api_key openai_api_key \
    --num_tasks 3
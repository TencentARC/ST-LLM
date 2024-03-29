export PYTHONPATH="./:$PYTHONPATH"
python stllm/test/gpt_evaluation/evaluate_activitynet_qa.py \
    --pred_path test_output/qabench/stllm_instructblipbase_msvdqa.json \
    --output_dir test_output/qabench/msvdQA/stllm_instructblipbase \
    --output_json test_output/qabench/msvdQA/stllm_instructblipbase/msvdQA.json \
    --api_key openai_api_key \
    --num_tasks 3
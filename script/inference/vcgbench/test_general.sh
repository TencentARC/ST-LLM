export PYTHONPATH="./:$PYTHONPATH"
python stllm/test/vcgbench/videochatgpt_benchmark_general.py \
    --cfg-path config/instructblipbase_stllm_conversation.yaml \
    --ckpt-path /Path/to/STLLM_conversation_weight \
    --video_dir /Path/to/video_chatgpt/Test_Videos \
    --gt_file /Path/to/video_chatgpt/Benchmarking_QA/generic_qa.json \
    --output_dir test_output/vcgbench/ \
    --output_name stllm_instructblipbase_general \
    --num-frames 64 \
    
    
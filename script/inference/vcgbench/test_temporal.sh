export PYTHONPATH="./:$PYTHONPATH"
python stllm/test/vcgbench/videochatgpt_benchmark_general.py \
    --cfg-path config/instructblipbase_stllm_conversation.yaml \
    --ckpt-path /Path/to/STLLM_conversation_weight \
    --video_dir /Path/to/video_chatgpt/Test_Videos \
    --gt_file /Path/to/Benchmarking_QA/temporal_qa.json \
    --output_dir test_output/vcgbench/ \
    --output_name stllm_instructblipbase_temporal \
    --num-frames 64 \
    
    
export PYTHONPATH="./:$PYTHONPATH"
python stllm/test/qabench/msrvtt_qa.py \
    --cfg-path config/instructblipbase_stllm_qa.yaml \
    --ckpt-path /Path/to/STLLM_QA_weight \
    --video_dir /Path/to/MSRVTT-QA/video/ \
    --gt_file /Path/to/MSRVTT-QA/test_qa.json \
    --output_dir test_output/qabench/ \
    --output_name stllm_instructblipbase_msrvttqa \
    --num-frames 64 \
    
    
    
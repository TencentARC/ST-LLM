export PYTHONPATH="./:$PYTHONPATH"
python script/inference/qabench/anet_qa.sh \
    --cfg-path config/instructblipbase_stllm_qa.yaml \
    --ckpt-path /Path/to/STLLM_QA_weight \
    --video_dir /Path/to/Anet/videos \
    --gt_file_question /Path/to/Anet/test_q.json \
    --gt_file_answers /Path/to/Anet/test_a.json \
    --output_dir test_output/qabench/ \
    --output_name stllm_instructblipbase_anetqa \
    --num-frames 16 \
    
    
    
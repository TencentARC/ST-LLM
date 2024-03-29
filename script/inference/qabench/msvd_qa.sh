export PYTHONPATH="./:$PYTHONPATH"
python stllm/test/qabench/msvd_qa.py \
    --cfg-path config/instructblipbase_stllm_qa.yaml \
    --ckpt-path /Path/to/STLLM_QA_weight \
    --video_dir /Path/to/MSVD/YouTubeClips \
    --gt_file /Path/to/MSVD-QA/test_qa.json \
    --output_dir test_output/qabench/ \
    --output_name stllm_instructblipbase_msvdqa \
    --num-frames 64 \

    
    
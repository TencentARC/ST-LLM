export PYTHONPATH="./:$PYTHONPATH"
python stllm/test/mvbench/mv_bench_infer.py \
    --cfg-path config/instructblipbase_stllm_qa.yaml \
    --ckpt-path Path/to/instructblipbase_stllm_qa \
    --anno-path Path/to/MVBench/json \
    --output_dir test_output/mvbench/ \
    --output_name instructblipbase_stllm_qa_mvbench_fps1 \
    --num-frames 0 \
    --ask_simple \
    
    


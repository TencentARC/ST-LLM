export PYTHONPATH="./:$PYTHONPATH"
deepspeed --master_port=20000 --include=localhost:0,1,2,3,4,5,6,7 stllm/train/train_hf.py --cfg-path /Path/to/desired/config
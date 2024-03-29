import argparse
import torch

from stllm.common.config import Config
from stllm.common.registry import registry
from stllm.conversation.conversation import Chat, CONV_instructblip_Vicuna0

# imports modules for registration
from stllm.datasets.builders import *
from stllm.models import *
from stllm.processors import *
from stllm.runners import *
from stllm.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='config/instructblipbase_stllm_conversation.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--ckpt-path", required=True, help="path to STLLM_conversation_weight.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

ckpt_path = args.ckpt_path
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = ckpt_path
model_config.llama_model = ckpt_path
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.to(torch.float16)
CONV_VISION = CONV_instructblip_Vicuna0

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

chat_state = CONV_VISION.copy()
video = 'example/BaoguoMa.mp4'
prompt = 'Tell me why this video looks so funny?'
img_list = []

chat.upload_video(video, chat_state, img_list, 64, text=prompt)
chat.ask("###Human: " + prompt + " ###Assistant: ", chat_state)
llm_message = chat.answer(conv=chat_state,
                img_list=img_list,
                num_beams=5,
                do_sample=False,
                temperature=1,
                max_new_tokens=300,
                max_length=2000)[0]
print (llm_message)



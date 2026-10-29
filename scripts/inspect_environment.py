import json, platform, sys
info={"python":sys.version,"platform":platform.platform()}
try:
 import torch
 info.update(torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),bf16_supported=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except ImportError: info["torch"]="not installed"
try:
 import transformers; info["transformers"]=transformers.__version__
except ImportError: info["transformers"]="not installed"
print(json.dumps(info,indent=2))

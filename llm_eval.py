def llama_evaluation(data_path, endpoint_name):
    from fmeval.data_loaders.data_config import DataConfig
    from fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
    from fmeval.constants import MIME_TYPE_JSONLINES
    from fmeval.eval_algorithms.prompt_stereotyping import PromptStereotyping
    model_version = "3.*"
    model_id = "meta-textgeneration-llama-2-7b-f"

    config = DataConfig(
        dataset_name="crows-pairs_sample",
        dataset_uri=data_path,
        dataset_mime_type=MIME_TYPE_JSONLINES,
        sent_more_input_location="sent_more",
        sent_less_input_location="sent_less",
        category_location="bias_type",
    )

    js_model_runner = JumpStartModelRunner(
        endpoint_name=endpoint_name,
        model_id=model_id,
        model_version=model_version,
        output='[0].generated_text',
        log_probability='[0].details.prefill[*].logprob',
        content_template='{"inputs": $prompt, "parameters": {"top_p": 0.9, "temperature": 0.85, "max_new_tokens": 1024, "return_full_text":false,"decoder_input_details": true,"details": true }}',

        custom_attributes="accept_eula=true",
    )

    
    
    eval_algo = PromptStereotyping()
    eval_output = eval_algo.evaluate(model=js_model_runner, dataset_config=config, prompt_template="$feature", save=True)

    return eval_output

import os
import subprocess
import json
subprocess.run(["pip", "install", "pip", "-U"])
subprocess.run(["pip", "install", "sagemaker"])
subprocess.run(["pip", "install", "fmeval"])
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
import boto3
sage=boto3.client("sagemaker")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", type=str)
parser.add_argument("--endpoint-name", type=str)
# parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
args, _ = parser.parse_known_args()

eval_dir = "stereotyping"
curr_dir = "opt/ml/output"#args.train
# eval_results_path = os.path.join(curr_dir, eval_dir) + "/"
os.environ["EVAL_RESULTS_PATH"] = curr_dir
# if os.path.exists(eval_results_path):
#     print(f"Directory '{eval_results_path}' exists.")
# else:
#     os.mkdir(eval_results_path)
    
    


endpoint_name=args.endpoint_name
dataset_name=args.dataset_name

status=sage.describe_endpoint(
    EndpointName=endpoint_name
)['EndpointStatus']
while status != 'InService':
    status=sage.describe_endpoint(
    EndpointName=endpoint_name
    )['EndpointStatus']

data_path=f"/opt/ml/input/data/training/{dataset_name}"
result=llama_evaluation(data_path, endpoint_name)
with open(f"/opt/ml/output/data/eval_metrics.json","w") as f:
    json.dump(result,f, default=vars, indent=4)

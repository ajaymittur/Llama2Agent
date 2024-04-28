import argparse
import os

from datasets import load_dataset
from datasets import concatenate_datasets

import sagemaker
from sagemaker.s3 import S3Uploader
from sagemaker import instance_types
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.jumpstart.model import JumpStartModel


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prep-dataset", 
                        help="Format the dataset to Llama 2 Chat template", 
                        type=bool,
                        default=False)
    group.add_argument("--train-data-s3",
                        help="Formatted training data in s3",
                        type=str)
    parser.add_argument("--epochs", 
                        help="Number of epochs to finetune Llama 2",
                        type=int,
                        default=5)
    parser.add_argument("--hf-dataset",
                        help="HuggingFace dataset of formatted trajectories",
                        type=str,
                        default="amittur/agent-instruct-llama2-chat")
    parser.add_argument("--model-uri",
                        help="Uri of custom model to fine tune. Can be used to resume finetuning from an earlier checkpoint",
                        type=str)
    parser.add_argument("--lora-r",
                        help="Rank of LoRA matrices",
                        type=int,
                        default=8)
    parser.add_argument("--int8",
                        help="Fine tune the quantized model (True/False)",
                        type=bool,
                        default=False)
    parser.add_argument("--max-input-length",
                        help="Max Input Length of Llama 2",
                        type=int,
                        default=1024)
    return parser.parse_args()


def conv_to_msg(conversation):
    messages = [{'role': 'system', 'content': 'You are a helpful, respectful and honest assistant.'}]

    for conv in conversation:
        messages.append({
            'role': 'user' if conv['from'] == 'human' else 'assistant',
            'content': conv['value']
        })

    return messages


if __name__ == '__main__':
    args = parse_args()
    train_data_location = args.train_data_s3

    if args.prep_dataset:
        dataset = load_dataset("THUDM/AgentInstruct")
        new_dataset = dataset.map(lambda row : {'dialog': conv_to_msg(row['conversations'])}, remove_columns=['id', 'conversations'])
        final_dataset = concatenate_datasets(new_dataset.values())
        
        HF_TOKEN = os.environ["HF_TOKEN"]
        final_dataset.push_to_hub(args.hf_dataset, token=HF_TOKEN)

        output_bucket = sagemaker.Session().default_bucket()
        local_data_file = "train.jsonl"
        train_data_location = f"s3://{output_bucket}/agentinstruct"
        S3Uploader.upload(local_data_file, train_data_location)
        print(f"Training data stored in s3 bucket: {train_data_location}")
    
    model_id = "meta-textgeneration-llama-2-7b-f"
    model_version = "3.*"

    instance_type = instance_types.retrieve_default(
        model_id=model_id,
        model_version=model_version,
        scope="training")
    print("Using instance type: ", instance_type)

    EXECUTION_ROLE = os.environ["SAGEMAKER_EXECUTION_ROLE"] 

    estimator = JumpStartEstimator(
        model_uri=args.model_uri,
        model_id=model_id,
        role=EXECUTION_ROLE,
        environment={"accept_eula": "true"},
        disable_output_compression=True,  
        instance_type="ml.g5.12xlarge",
    )

    estimator.set_hyperparameters(
        chat_dataset="True", 
        instruction_tuned="False", 
        epoch=args.epochs, 
        max_input_length=args.max_input_length, 
        lora_r=args.lora_r,
        int8_quantization=args.int8
    )
    estimator.fit({"training": train_data_location})

    print("Finished fine tuning. Model stored in:", estimator.model_data)

    print("Deploying finetuned model...")
    predictor = estimator.deploy()
    print("Model deployed at:", predictor.endpoint)
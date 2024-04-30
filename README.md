# Enabling Large Language Models as Intelligent Agents with Llama 2

This repository contains our own tree-of-thought implementation of Llama 2 7B Chat on [AgentBench](https://github.com/THUDM/AgentBench) environments and our LoRA instruction finetuning script on sagemaker.

## Dataset
We use the [AgentInstruct](https://huggingface.co/datasets/THUDM/AgentInstruct) dataset and format the data into the [ReAct](https://arxiv.org/abs/2210.03629) format before instruction tuning Llama 2 7B Chat. The dataset contains high quality trajectories derived from GPT-4 for the following environments:

-  Operating System (OS)
-  Database (DB)
-  Knowledge Graph (KG)
-  Digital Card Game (DCG)
-  Lateral Thinking Puzzles (LTP)
-  [ALFWorld](https://github.com/alfworld/alfworld))
-  ([WebShop](https://github.com/princeton-nlp/webshop))
-  ([Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web))

The train and test splits for each of the environments are shown in the [original paper](https://arxiv.org/abs/2308.03688):
![dataset](./assets/statistics.png)

## Steps to Run
**Prerequisites** You will need to have [Docker](https://www.docker.com/) and [Python](https://python.org) installed on your computer/virtual machine

### Getting setup
1. Create a python virtual environment `python -m venv env`
2. Activate the virtual environment `source env/bin/activate`
3. Install the required packages `pip install -r requirements.txt`

### To run evaluations
1. Uncomment the environments you want to run in `configs/default.yaml` and `start_task.yaml`
2. Choose the agent you want to evaluate by uncommenting the agent name in `configs/default.yaml`
3. Deploy the model on sagemaker and update its endpoint in the agent config file `config/agents/<agent>.yaml`
4. Start the task assigner with `python -m src.start_task -a`
5. In another terminal window start the agents to evaluate with `python -m src.assigner`

### Fine-tuning Llama 2 7B Chat
1. Setup AWS credentials on the machine
2. Create a sagemaker domain and user
3. Run the finetuning script `python finetune-llama2-agent.py` see help for arguments
4. You should see a training job created and running on sagemaker

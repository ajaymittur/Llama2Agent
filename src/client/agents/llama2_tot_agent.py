from copy import deepcopy   
import re
from typing import List

from sagemaker.predictor import Predictor
from sagemaker.base_serializers import JSONSerializer
from sagemaker.base_deserializers import JSONDeserializer

from ..agent import AgentClient


class Llama2ToTAgent(AgentClient):
    def __init__(self, endpoint_name, branching_factor=3, *args, **config):
        super().__init__(*args, **config)
        
        JSON_SERIALIZER = JSONSerializer()
        JSON_DESERIALIZER = JSONDeserializer()

        self.branching_factor = branching_factor
        self.parameters = config
        self.predictor = Predictor(endpoint_name, serializer=JSON_SERIALIZER, deserializer=JSON_DESERIALIZER)
    
    def format_messages(self, messages: List[dict]) -> List[str]:
        """Format messages for Llama-2 chat models.
        
        The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and 
        alternating (u/a/u/a/u...). The last message must be from 'user'.
        """
        prompt: List[str] = []

        if messages[0]["role"] == "system":
            content = "".join(["<<SYS>>\n", messages[0]["content"], "\n<</SYS>>\n\n", messages[1]["content"]])
            messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]

        for user, answer in zip(messages[::2], messages[1::2]):
            prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])

        prompt.extend(["<s>", "[INST] ", (messages[-1]["content"]).strip(), " [/INST] "])

        return "".join(prompt)

    def llama2(self, history: List[dict]) -> str:
        prompt = self.format_messages(history)
        payload = {'inputs': prompt, 'parameters': self.parameters}
        response = self.predictor.predict(payload)
        return response[0]["generated_text"]

    def rollout(self, history: List[dict]) -> List[str]:
        """Perform action rollout in tree-of-thought""" 
        propose_prompt = f"Propose {self.branching_factor} possible actions to take next. Remember to follow the initial task instructions. List the actions like this:"
        propose_prompt += ' ' + '\n'.join(["Action: <action>"] * self.branching_factor)
        branch_convo = deepcopy(history)
        branch_convo[-1]['content'] += f"\n{propose_prompt}"
        output = self.llama2(branch_convo)
        next_actions = re.findall(r'Action: (.+)', output) 
        return next_actions
    
    def critic(self, history: List[dict], actions: List[str]) -> List[int]:
        value_prompt = """Assign a numerical value to the action.
The value should range from 1 to 3 where:
1 - The action makes little to no progress towards goal or is completely wrong
2 - The action is in the right direction, but may not help with achieving the goal
3 - The action is an Answer or the Operation is correct and helps solve a subgoal or provides information that informs your next action

Action: {action}
Follow the following format, do not including anything but the value in your response:
Value: <value>
"""
        value_convo = deepcopy(history)
        action_scores = []
        for act in actions:
            value_convo[-1]['content'] += "\n" + value_prompt.format(action=act)
            output = self.llama2(value_convo)
            value = re.search(r'Value: (\d+)', output).group(1)
            action_scores.append(int(value))
        return action_scores

    def get_thought(self, history: List[dict], action: str) -> str:
        thought_prompt = f"""You decide to take the following action:
Action: {action}

Justify the thought behind taking this action.
Thought: 
"""
        thought_convo = deepcopy(history)
        thought_convo[-1]['content'] += f"\n{thought_prompt}"
        thought = self.llama2(thought_convo)
        return thought

    def inference(self, history: List[dict]) -> str:
        candidate_actions = self.rollout(history)
        critic_scores = self.critic(history, candidate_actions)
        best_action, best_score = 0, 0
        for i, score in enumerate(critic_scores):
            if score > best_score:
                best_action = i
                best_score = score
        thought = self.get_thought(history, candidate_actions[best_action])
        result = f"""Thought: {thought}
Action: {candidate_actions[best_action]}
"""
        return result

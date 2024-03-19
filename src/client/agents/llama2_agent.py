import os
from copy import deepcopy
from typing import List

from sagemaker.predictor import Predictor
from sagemaker.base_serializers import JSONSerializer
from sagemaker.base_deserializers import JSONDeserializer

from ..agent import AgentClient


class Llama2Agent(AgentClient):
    def __init__(self, endpoint_name, *args, **config):
        super().__init__(*args, **config)
        
        JSON_SERIALIZER = JSONSerializer()
        JSON_DESERIALIZER = JSONDeserializer()

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

    def inference(self, history: List[dict]) -> str:
        prompt = self.format_messages(history)
        payload = {'inputs': prompt, 'parameters': self.parameters}
        response = self.predictor.predict(payload)
        return response[0]["generated_text"]

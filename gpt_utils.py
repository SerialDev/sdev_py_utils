
import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = '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'

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-0301",
    max_tokens=100,
  messages=[
        {"role": "system", "content":
         '''You are a helpful assistant that takes inspiration from world class developers like
         John Carmack, Casey Muratori and, Jonathan Blow, you will give me code recommendations heavily influenced
         by data oriented design maxims, and high quality performant code '''},
        {"role": "user", "content": "Can you show me a good example of code"},
    ]
)

dir(response)
response.values()

def test():
    pass

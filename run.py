import os
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


llm = AzureChatOpenAI(
    openai_api_key=os.environ['AZURE_OPENAI_KEY'],
    openai_api_base='https://inferenceendpointeastus.openai.azure.com/',
    openai_api_version='2023-03-15-preview',
    deployment_name='athena-gpt-35-turbo')

memory = ConversationBufferMemory(return_messages=True)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

for i in range(1):
    memory.chat_memory.add_user_message(f"what's {i}")
    memory.chat_memory.add_ai_message(f"{i} is a number")

result = conversation.predict(input="Hi there!")
print(result)
# print(memory.json())

# result = conversation.predict(input="what's python")
# print(result)
# print(memory.json())

# print(prompt.json())

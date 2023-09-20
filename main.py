from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate

# import POCScript
from Conversation import LLMModelClass

app = FastAPI(debug=True)
qa_chain = LLMModelClass()

qa_chain_object = qa_chain.loadLLMmodel()


# Create a class for the input data
class InputData(BaseModel):
    prompt: str


# Create a class for the output dat
class OutputData(BaseModel):
    response: str


# Create a route for the web application
def generateResponseFromLLMObject(prompt):
    response = qa_chain_object({"question": prompt})
    return response


@app.post("/generate")
async def generate(request: Request, input_data: InputData):
    prompt = input_data.prompt
    print("Prompt received:" + prompt)
    response = generateResponseFromLLMObject(prompt)
    # response = qa_chain_object({"question": prompt})
    # response = POCScript.llm_model(prompt)
    # print("Response sent:" + response)
    return {"response": response}


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
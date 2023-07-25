#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain.llms.base import LLM
from langchain.llms import OpenAI
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import os
import argparse
import time

class ChatGLM(LLM):
    max_token: int = 2048
    temperature: float = 0.01
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 10
    
    def __init__(self):
        super().__init__()
        
    @property
    def _llm_type(self) -> str:
        return "ChatGLM"
            
    def load_model(self, model_path=None):
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, config=model_config, trust_remote_code=True).to("cpu", dtype=float)

    def _call(self,prompt:str,history:List[str] = [],stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
                    self.tokenizer,prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,temperature=self.temperature,
                    top_p=self.top_p)
        return response

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, temperature=0.0, n_ctx=8192, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "ChatGLM":
            llm = ChatGLM()
            llm.load_model(model_path)
        case "OpenAI":
            os.environ["OPENAI_API_KEY"] = ""
            os.environ["OPENAI_API_BASE"] = "https://oa.api2d.net/v1"
            llm = OpenAI(temperature=0, max_tokens=512)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All, ChatGLM, OpenAI")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()

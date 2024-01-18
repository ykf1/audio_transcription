from dotenv import load_dotenv
import tiktoken

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.runnables import RunnablePassthrough

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains.summarize import load_summarize_chain
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

model = "gpt-3.5-turbo"

class LangchainHelper():

    def __init__(self, transcript: str):
        
        self.llm = ChatOpenAI(temperature=0, model_name=model)
        self.documents = self.create_documents(transcript)
        self.retriever = self.create_embeddings()
        self.qna_chain = self.create_qna_chain()
        self.summary_chain = self.create_summary_chain(transcript)


    def create_documents(self, transcript: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100,
            length_function = len,
            separators=["\n\n", "\n", "."]
        )
        return text_splitter.create_documents([transcript])


    def create_embeddings(self):
        persist_directory = 'vectordb'
        embedding = OpenAIEmbeddings(model='text-embedding-ada-002')
        vectordb = Chroma.from_documents(
            documents=self.documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        # vectordb = Chroma(
        #     embedding_function=embedding,
        #     persist_directory=persist_directory
        # )

        #vectordb.persist()
        return vectordb.as_retriever()


    def create_summary_chain(self, transcript: str):
        """Creates summary chain. If transcript has token count within context window, use stuff for chain type. Else, use map reduce. Returns the chain object."""
        
        num_tokens: int = self.num_tokens_from_string(transcript)
        
        if num_tokens < 4000:
            
            summarise_template = """You are provided with a transcript of people having a conversation.
                Write a summary that will highlight the key points mentioned.
                Do not respond with anything outside of the transcript. If you don't know, say 'I don't know'

                Respond with the following format:
                {output_format}

                Transcript:
                {text}
            """
            
            chain = load_summarize_chain(
                self.llm, 
                chain_type="stuff",
                prompt=self.create_prompt_from_template(summarise_template),
                verbose=False
            )

        else:
            map_template = """You are provided with a transcript of people having a conversation.
                Write a summary that will highlight the key points mentioned.
                Do not respond with anything outside of the transcript. If you don't know, say 'I don't know'

                Transcript:
                {text}
            """
            combine_template = """You are provided with a transcript of people having a conversation.
                Write a summary that will highlight the key points mentioned.
                Do not respond with anything outside of the transcript. If you don't know, say 'I don't know'

                You must respond with your answer in the following format:
                {output_format}

                Transcript:
                {text}
            """

            chain = load_summarize_chain(
                self.llm, 
                chain_type="map_reduce",
                map_prompt=self.create_prompt_from_template(map_template),
                combine_prompt=self.create_prompt_from_template(combine_template),
                verbose=True
            )
        return chain


    def summarise(self, user_output_choice: str) -> str:
        summary_output_options = {
            "one sentence": "Only one sentence",
            "bullet points": "Bullet point format where each bullet point should be consise",
            "short": "A few short sentences, not longer than 5 sentences.",
            "long": "A verbose summary. You may do a few paragraphs to describe the transcript if needed."
        }
        
        response = self.summary_chain.invoke(
            {
                "input_documents": self.documents,
                "output_format": summary_output_options[user_output_choice],
            }
        )
        return response['output_text']


    def create_qna_chain(self):
        qna_template = """Answer the question based only on the following context. \
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer. \
        If the question is not related to the context, politely respond that you are tuned to only answer \
        questions that are related to the context. Use as much detail as possible when responding. \

        Context:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(qna_template)

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain


    def get_response(self, prompt: str) -> str:
        return self.qna_chain.invoke(prompt)
    

    def create_prompt_from_template(self, template: str) -> ChatPromptTemplate:
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        return ChatPromptTemplate.from_messages(messages=[system_message_prompt])


    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(string))


if __name__ == "__main__":
    pass

from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any

class ArticleSummary(BaseModel):
    title: str = Field(description="Article's title based on the content, unbiased, without spin or clickbait")
    summary: str = Field(description="Summary, one paragraph summary of the story. Do not preface it with 'this story discusses...' or any other introduction.")
    time: datetime = Field(description="The date and time of the story")
    importance: int = Field(description='''Importance, the story importance on 1 to 10 scale, where 10 is the most important. Anything ranked 10 would represent
        immediate life-threatening or global crisis like war, pandemics, or climate disasters. 1 is the story of no importance to anyone.''')
    keywords: List[str] = Field(description="The list of keywords for this story. If a company is mentioned, include the company name as a keyword.")
    category: str = Field(description="The news category of the story")

def summarize_article(article: str) -> dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000)
    sys_prompt = '''
You are an expert journalist capable of analyzing news stories in depth.
'''
    user_prompt = '''
        Analyze the following news story and return information about it in json format, with the following fields:

        {format_instructions}

        The story follows:
        {article}
    '''

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])

    parser = PydanticOutputParser(pydantic_object=ArticleSummary)

    res = llm.invoke(prompt_template.format_messages(article=article, format_instructions=parser.get_format_instructions()))
    return parser.parse(res.content).model_dump()

def get_text_embeddings(text: str, model: str = 'text-embedding-ada-002', dimensions: int = 1536) -> List[float]:
    """
    Get embeddings for a given text using OpenAI API.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        list: A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If there is an error generating embeddings.
    """
    embeddings = OpenAIEmbeddings(model=model, dimensions=dimensions)
    return embeddings.embed_query(text)

import json
from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings import OpenAIEmbeddings

class ArticleSummary(TypedDict):
    summary: str
    time: str
    importance: int  # 1-10
    keywords: List[str]
    category: str

def summarize_article(article: str) -> ArticleSummary:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000)
    sys_prompt = '''
You are an expert journalist capable of analyzing news stories in depth.
'''
    user_prompt = '''
 Analyze the following news story and return information about it in json format, with the following fields:

* summary, one paragraph summary of the story. Do not preface it with "this story discusses..." or any other introduction.
* time, the date and time of the story
* importance, the story importance on 1 to 10 scale, where 10 is the most important. Anything ranked 10 would represent
immediate life-threatening or global crisis like war, pandemics, or climate disasters. 1 is the story of no importance to anyone.
* keywords, a list of keywords for this story. If a company is mentioned, include the company name as a keyword.
* category, the news category of the story.

Return only the json as described above and nothing else.

The story follows:
{article}
'''
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])

    res = llm.invoke(prompt_template.format_messages(article=article))
    return json.loads(res.content)

def get_text_embeddings(text: str) -> List[float]:
    """
    Get embeddings for a given text using OpenAI API.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        list: A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If there is an error generating embeddings.
    """
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_query(text)

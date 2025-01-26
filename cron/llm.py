import os
from openai import OpenAI
import json
from typing import List, Optional, TypedDict, Literal

class ArticleSummary(TypedDict):
    summary: str
    time: str
    importance: int  # 1-10
    keywords: List[str]
    category: str

def summarize_article(article: str) -> ArticleSummary:
    """
    Summarize an article using OpenAI API.

    Args:
        article (str): The article to summarize.

    Returns:
        ArticleSummary: A dictionary with typed fields for summary, time, importance, keywords and category.

    Raises:
        RuntimeError: If there is an error processing the article with OpenAI.
        ValueError: If the OpenAI response is missing required fields or has invalid values.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "you are an expert journalist capable of analyzing news stories in depth "},
                {"role": "user", "content": f'''
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
                 '''}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        res = response.choices[0].message.content
        res = res.replace("```json", "").replace("```", "")
        data = json.loads(res)
        
        # Validate response has required fields and correct types
        if not all(k in data for k in ArticleSummary.__annotations__):
            missing = set(ArticleSummary.__annotations__) - set(data)
            raise ValueError(f"Response missing required fields: {missing}")
        
        if not isinstance(data['importance'], int) or not 1 <= data['importance'] <= 10:
            raise ValueError(f"Importance must be int between 1-10, got: {data['importance']}")
            
        if not isinstance(data['keywords'], list):
            raise ValueError(f"Keywords must be a list, got: {type(data['keywords'])}")
            
        return data
    except Exception as e:
        raise RuntimeError(f"Error processing with OpenAI: {e}")

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
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            input=text,
            model='text-embedding-ada-002'
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings: {e}")

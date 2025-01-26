import os
from openai import OpenAI
import json

openai_api_key = os.getenv("OPENAI_API_KEY")

def summarize_article(article):
    """
    Summarize an article using OpenAI API.

    Args:
        article (str): The article to summarize.

    Returns:
        dict: A dictionary with the summary, time, importance, and keywords of the article.
    """
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
        return json.loads(res)
    except Exception as e:
        return f"Error processing with OpenAI: {e}"

def get_text_embeddings(text):
    """
    Get embeddings for a given text using OpenAI API.

    Args:
        api_key (str): Your OpenAI API key.
        model (str): The embedding model to use (e.g., 'text-embedding-ada-002').
        text (str): The text to generate embeddings for.

    Returns:
        list: A list of floats representing the embedding vector.
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            input=text,
            model='text-embedding-ada-002'
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error: {e}")
        return None

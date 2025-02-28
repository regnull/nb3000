import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import wikipedia

model = ChatOpenAI(model="gpt-4o-mini")

def anaylize_place(keyword: str) -> any:
    return wikipedia.summary(keyword, sentences=10)

def analyze_keyword(keyword: str) -> any:
    tagging_prompt = ChatPromptTemplate.from_template(
    """
    You are give a keyword. Extract the desired information about the keyword.

    Only extract the properties mentioned in the 'Classification' function.

    Keyword:
    {keyword}
    """
    )

    class Classification(BaseModel):
        proper_noun: bool = Field(description="Is the keyword a proper noun?")
        obscure: bool = Field(
            description="Does the keyword refer to an obscure term or entity?"
        )
        is_person: bool = Field(
            description="Does the keyword refer to a person?"
        )
        is_place: bool = Field(
            description="Does the keyword refer to a place?"
        )
        is_thing: bool = Field(
            description="Does the keyword refer to a thing?"
        )
        is_abstract: bool = Field(
            description="Does the keyword refer to an abstract concept?"
        )
        is_organization: bool = Field(
            description="Does the keyword refer to an organization?"
        )


    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
        Classification
    )

    response = llm.invoke(tagging_prompt.invoke({"keyword": keyword}))
    return response

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '../.env')
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)

    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_database('nb3000')
    keywords_col = db.get_collection('keywords')
    
    keywords = keywords_col.find({'analyzed': {'$exists': False}})
    for k in keywords:
        print(f"Analyzing {k['keyword']}")
        r = analyze_keyword(k['keyword'])
        info = {
            'analyzed': datetime.now(),
            'analysis': r.model_dump()
        }
        if r.proper_noun:
            try:
                p = wikipedia.page(k['keyword'])
                info['wikipedia'] = {
                    'summary': p.summary,
                    'url': p.url,
                    'image_url': p.images[0] if p.images else None
                }
            except wikipedia.exceptions.PageError as e:
                print(f"PageError: {e}")
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"DisambiguationError: {e}")
                
        keywords_col.update_one({'_id': k['_id']}, {'$set': info})
    
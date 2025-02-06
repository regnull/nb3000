from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch
from pydantic import BaseModel, Field
from pymongo import MongoClient
import os

model = ChatOpenAI(model="gpt-4o-mini")

class Classification(BaseModel):
    is_proper_noun: bool = Field(description="Is the keyword a proper noun?")
    is_obscure: bool = Field(
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


wikipedia = WikipediaAPIWrapper()

def wikipedia_search(inputs):
    classification = inputs["classification"]
    if not classification.is_place:
        return {"wikipedia_summary": None}

    query = inputs['original_input']['keyword']
    summary = wikipedia.run(query)
    return {"wikipedia_summary": summary}

def analyze_keyword(keyword: str) -> str:
    tagging_prompt = ChatPromptTemplate.from_template(
    """
    You are give a keyword. Extract the desired information about the keyword.

    Only extract the properties mentioned in the 'Classification' function.

    Keyword:
    {keyword}
    """
    )
    llm = ChatOpenAI(
        temperature=0,
        # output_key="classification",
        model="gpt-4o-mini").with_structured_output(Classification)
   
    
    analyze_chain = LLMChain(llm=llm, prompt=tagging_prompt)
    
    branch = RunnableBranch(
       (lambda input: input['wikipedia_search'] is not None, wikipedia_search),  # If the condition is True, run the LLM chain
        default_runnable  # If the condition is False, run the default action
    )
    
    chain = (
        {
            'original_input': RunnablePassthrough(),
            'classification': tagging_prompt | llm
        }
        | RunnableBranch(
            (lambda input: input['wikipedia_search'] is not None, wikipedia_search),  # If the condition is True, run the LLM chain
            default_runnable  # If the condition is False, run the default action
        )
        # | RunnableLambda(wikipedia_search)
    )
    
    response = chain.invoke({"keyword": keyword})
    return response

if __name__ == "__main__":
    result = analyze_keyword("West Bank")
    print(result)
    exit()
    
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client.get_database('nb3000')
    keywords_col = db.get_collection('keywords')
    for keyword in keywords_col.find():
        print(f"Analyzing {keyword['keyword']}")
        if keyword.get('analysis'):
            print(f"Skipping {keyword['keyword']} because it already has an analysis")
            continue

        analysis = analyze_keyword(keyword['keyword'])
        keywords_col.update_one(
            { "_id": keyword["_id"] },
            { "$set": { "analysis": analysis.dict() } }
        )

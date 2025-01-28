from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

model = ChatOpenAI(model="gpt-4o-mini")

def analyze_keyword(keyword: str) -> str:
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
    keywords = [
        "Rebels",
        "Congo",
        "Rwanda",
        "peacekeepers",
        "diplomatic ties",
        "music",
        "documentary",
        "Baltic Sea",
        "Latvia",
        "Trump",
        "Colombia",
        "Tariffs",
        "Amorphophallus gigas",
        "Homo juluensis",
        "La Liga",
        "Proud Boys",
        "Trump",
        "Sviatlana Tsikhanouskaya",
    ]
    for keyword in keywords:
        print(f"{keyword}: {analyze_keyword(keyword)}")

import pytz
from typing import List, Any, Union, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json # For formatting input to LLMs if needed

class ArticleSummary(BaseModel):
    title: str = Field(description="Article's title based on the content, unbiased, without spin or clickbait")
    summary: str = Field(description='''Summary, one paragraph summary of the story. Do not preface it with 
                         'this story discusses...' or any other introduction. Never refer to 'the story' or 
                         'the article' in the summary, just describe the content.''')
    time: datetime = Field(description="The date and time of the story")
    importance: int = Field(description='''Assign importance on a 1 to 10 scale, based on how the world would be affected by the news. 
Global crises (e.g., war, pandemics, major natural disasters, significant political or economic shifts) should receive the highest importance (9-10). 
News primarily concerning celebrities, sports, or localized non-critical events should receive lower importance (1-3). 
Consider the breadth and depth of impact. 10 represents an immediate, life-threatening global crisis.''')
    keywords: List[str] = Field(description="The list of keywords for this story. If a company is mentioned, include the company name as a keyword.")
    category: str = Field(description="The news category of the story")
    language: str = Field(description="The language of the story")

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
    parsed_data = parser.parse(res.content).model_dump()
    return parsed_data

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
    if model == 'text-embedding-3-small':
        embeddings = OpenAIEmbeddings(model=model, dimensions=dimensions)
    else:
        embeddings = OpenAIEmbeddings(model=model)
    return embeddings.embed_query(text)

def summarize_stories(stories: list[dict]) -> dict:
    # Sort stories by date (most recent first)
    sorted_stories = sorted(
        stories,
        key=lambda story: (lambda dt_val: dt_val.replace(tzinfo=pytz.UTC) if isinstance(dt_val, datetime) and dt_val.tzinfo is None else dt_val)(story.get('updated', datetime.min.replace(tzinfo=pytz.UTC))),
        reverse=True
    )

    articles = "\n\n".join(
        f"ARTICLE {i+1}:\n{s['updated'].strftime('%Y-%m-%d %H:%M:%S')}\n{s['headline']}\n{s['summary']['summary']}"
        for i, s in enumerate(sorted_stories)
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000)
    sys_prompt = '''
You are an expert journalist capable of analyzing news stories in depth.
'''
    user_prompt = '''
        You are given several articles on the same subject. The articles are sorted by recency,
        with the most recent article first. Your job is to summarize the articles and return
        the information in the format described below.
        The time of the summary must be the time of the most recent article.

        {format_instructions}

        The articles follow:
        {articles}
    '''
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])

    parser = PydanticOutputParser(pydantic_object=ArticleSummary)

    res = llm.invoke(prompt_template.format_messages(articles=articles, format_instructions=parser.get_format_instructions()))
    parsed_data = parser.parse(res.content).model_dump()
    
    print("\n\nSummarizing stories:")
    print(articles)
    print("\nSummarized stories:")
    print(parsed_data)
    
    return parsed_data

# --- Pydantic Models for Daily Summary Workflow (Revised with Short Names) ---

class InitialDailySummaryOutput(BaseModel):
    date: datetime = Field(description="The date for which this news summary is generated.")
    title: str = Field(description="A concise, engaging title for the day's news summary.")
    plain_text_summary: str = Field(description="A comprehensive one-page PLAIN TEXT summary of the most significant news from the past 24 hours. No HTML links.")
    top_keywords: List[str] = Field(description="A list of 5-7 most prominent keywords or key phrases.")
    key_story_titles: List[str] = Field(description="Headlines of 3-5 key stories mentioned.")
    sentiment: str = Field(description="Overall sentiment (e.g., Positive, Negative, Neutral, Mixed).")

class TopicShortNameOutput(BaseModel):
    short_name: str = Field(description="A very concise (2-5 words) and unique key phrase or short name for the topic, suitable for an LLM to recognize later when scanning text.")

class PlainParagraphsOutput(BaseModel):
    paragraphs: List[str] = Field(description="A list of strings, where each string is a logically separated paragraph of plain text.")

class TextChunk(BaseModel):
    type: str = Field(default="text")
    content: str

class PotentialLinkChunk(BaseModel):
    type: str = Field(default="potential_link")
    identified_short_name: str = Field(description="The short_name of the topic identified in this text segment.")
    link_text: str = Field(description="The actual segment of text from the paragraph that corresponds to this topic and should become the link text.")

class SegmentedParagraphWithShortNamesOutput(BaseModel):
    chunks: List[Union[TextChunk, PotentialLinkChunk]] = Field(description="A list of text and link chunks for a single paragraph.")

class LinkableTopicInfo(BaseModel):
    topic_id: str
    title: str
    summary: str 

# Final structure for MongoDB
class DailyNewsSummary(BaseModel):
    date: datetime 
    title: str 
    overall_summary: str # This will contain the final HTML with <p> tags and <a> tags
    top_keywords: List[str] 
    key_story_titles: List[str] 
    sentiment: str 

# --- LLM Functions for Daily Summary Workflow (Revised with Short Names) ---

def generate_initial_daily_summary(articles_data: List[dict], llm_model: str = "gpt-4o-mini") -> dict:
    llm = ChatOpenAI(model=llm_model, temperature=0.7, max_tokens=2000)
    sys_prompt = '''
    You are an expert news editor. Your task is to create a comprehensive yet concise one-page PLAIN TEXT summary 
    of the day's most important news based on the provided articles. 
    The summary should be well-organized, easy to read, and cover a diverse range of significant events. 
    Focus on clarity, accuracy, and an objective tone. 
    ABSOLUTELY DO NOT include any HTML formatting or hyperlinks in the 'plain_text_summary' field.
    The output MUST be a valid JSON object.
    '''
    user_prompt = '''
    Based on the following collection of news articles (each with a headline and summary) from the past 24 hours, 
    please generate a daily news briefing. Respond with a JSON object that strictly adheres to the following format instructions.
    The 'plain_text_summary' field should contain only plain text, be approximately 500-700 words, and have no HTML.
    Ensure the 'date' field in your JSON output reflects today's date.

    {format_instructions}

    Article Data:
    {articles_input_str}
    '''

    articles_input_parts = []
    for i, article in enumerate(articles_data):
        part = f"--- ARTICLE {i+1} ---\n"
        part += f"HEADLINE: {article.get('headline', 'N/A')}\n"
        part += f"SUMMARY: {article.get('summary_text', 'N/A')}\n\n"
        articles_input_parts.append(part)
    articles_input_str = "".join(articles_input_parts)

    parser = PydanticOutputParser(pydantic_object=InitialDailySummaryOutput)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])

    res = llm.invoke(prompt_template.format_messages(articles_input_str=articles_input_str, format_instructions=parser.get_format_instructions()))
    parsed_data = parser.parse(res.content).model_dump()
    # Override date to be sure, as LLMs can sometimes pick a date from the articles
    parsed_data['date'] = datetime.now(pytz.utc)
    return parsed_data

def structure_plain_text_into_paragraphs(plain_text_summary: str, llm_model: str = "gpt-4o-mini") -> dict:
    llm = ChatOpenAI(model=llm_model, temperature=0.3, max_tokens=2500) # Max tokens might need adjustment
    sys_prompt = '''
    You are an expert content structurer. Your task is to take a single block of plain text (a news summary) 
    and break it down into a list of strings, where each string represents a logically separated paragraph. 
    The goal is to improve readability. Paragraphs should not be too short unless it is a single, impactful statement. 
    Aim for thoughtful paragraph breaks that group related ideas. Preserve the original wording and casing.
    The output MUST be a valid JSON object that strictly follows the Pydantic model for PlainParagraphsOutput (a list of strings under a "paragraphs" key).
    '''
    user_prompt = '''
    Please process the 'PLAIN TEXT NEWS SUMMARY' provided below. 
    Segment it into a list of strings, where each string is a paragraph.
    Respond with a JSON object adhering to the Pydantic format instructions for 'PlainParagraphsOutput'.

    {format_instructions}

    PLAIN TEXT NEWS SUMMARY:
    {plain_text_summary}
    '''
    parser = PydanticOutputParser(pydantic_object=PlainParagraphsOutput)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])
    res = llm.invoke(prompt_template.format_messages(
        plain_text_summary=plain_text_summary,
        format_instructions=parser.get_format_instructions()
    ))
    parsed_data = parser.parse(res.content).model_dump()
    return parsed_data

def segment_paragraph_for_short_name_linking(paragraph_text: str, enriched_linkable_topics: List[Dict[str, str]], llm_model: str = "gpt-4o-mini") -> dict:
    llm = ChatOpenAI(model=llm_model, temperature=0.2, max_tokens=1500)
    sys_prompt = '''
    You are an expert text processing system. Your task is to analyze a single paragraph of a news summary and a list of known topics (each with a 'topic_id', 'title', 'summary', and unique 'short_name').
    Segment the paragraph into an ordered sequence of chunks: 'text' chunks or 'potential_link' chunks.
    - If a segment of the paragraph clearly and directly corresponds to one of the topics provided in 'ENRICHED LINKABLE TOPICS DATA' (match based on the topic's 'short_name', using its 'title' and 'summary' for context and confirmation), 
      mark it as a 'potential_link' chunk. This chunk must include the 'identified_short_name' of the matched topic and the 'link_text' (the actual text segment from the paragraph that refers to this topic).
      Only identify a topic if the match to its 'short_name' and context is strong. Do not force matches.
    - All other segments are 'text' chunks; provide their 'content'.
    The output MUST be a valid JSON object strictly following the Pydantic model for SegmentedParagraphWithShortNamesOutput.
    Preserve original casing, spacing, and ALL PUNCTUATION (including sentence-ending periods) from the input paragraph across the generated chunks.
    If a sentence ends with text that becomes 'link_text', that 'link_text' MUST include the original sentence-ending punctuation.
    '''
    user_prompt = '''
    Process the 'PARAGRAPH TEXT' below. Segment it into 'text' and 'potential_link' chunks based on the 'ENRICHED LINKABLE TOPICS DATA'.
    For 'potential_link' chunks, provide the 'identified_short_name' and 'link_text'. For 'text' chunks, provide 'content'.
    Ensure original punctuation is preserved in chunk text. Respond with JSON as per format instructions.

    {format_instructions}

    ENRICHED LINKABLE TOPICS DATA (each has topic_id, title, summary, short_name):
    {linkable_topics_str}

    PARAGRAPH TEXT:
    {paragraph_text}
    '''
    
    linkable_topics_parts = []
    if enriched_linkable_topics:
        for i, topic_info in enumerate(enriched_linkable_topics):
            part = f"--- TOPIC ENTRY {i+1} ---\n"
            part += f"SHORT_NAME: {topic_info.get('short_name')}\n"
            part += f"TOPIC_ID (for your reference, do not output): {topic_info.get('topic_id')}\n"
            part += f"TITLE (context): {topic_info.get('title')}\n"
            part += f"SUMMARY (context): {topic_info.get('summary')}\n\n"
            linkable_topics_parts.append(part)
    linkable_topics_str = "".join(linkable_topics_parts) if linkable_topics_parts else "No specific topics provided for linking."

    parser = PydanticOutputParser(pydantic_object=SegmentedParagraphWithShortNamesOutput)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])
    res = llm.invoke(prompt_template.format_messages(
        paragraph_text=paragraph_text,
        linkable_topics_str=linkable_topics_str,
        format_instructions=parser.get_format_instructions()
    ))
    parsed_data = parser.parse(res.content).model_dump()
    return parsed_data

# NEW LLM Function (LLM 2a) to generate a short name for a topic
def generate_topic_short_name(topic_title: str, topic_summary: str, llm_model: str = "gpt-4o-mini") -> dict:
    llm = ChatOpenAI(model=llm_model, temperature=0.3, max_tokens=50)
    sys_prompt = '''
    You are a concise content analyst. Given a topic title and summary, generate a very short (2-5 words), 
    unique, and descriptive key phrase or "short name" for this topic. This short name will be used by another AI 
    to identify mentions of this topic in a broader text. It should be distinctive.
    Example: For title "Global Economic Summit Addresses Inflation Concerns" and a relevant summary, a good short name might be "Global Inflation Summit".
    Another Example: Title "New Discoveries on Mars Rover Mission", short name: "Mars Rover Discoveries".
    The output MUST be a valid JSON object strictly following the Pydantic model for TopicShortNameOutput.
    '''
    user_prompt = '''
    Generate a unique and descriptive short name (2-5 words) for the following topic:
    TITLE: {topic_title}
    SUMMARY: {topic_summary}

    {format_instructions}
    '''
    parser = PydanticOutputParser(pydantic_object=TopicShortNameOutput)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])
    res = llm.invoke(prompt_template.format_messages(
        topic_title=topic_title,
        topic_summary=topic_summary,
        format_instructions=parser.get_format_instructions()
    ))
    parsed_data = parser.parse(res.content).model_dump()
    return parsed_data

# --- Pydantic Models for Simplified Daily Summary Workflow ---

class SimplifiedDailySummaryOutput(BaseModel):
    date: datetime = Field(description="The date for which this news summary is generated.")
    title: str = Field(description="A concise, engaging title for the day's news summary.")
    paragraphed_summary: str = Field(description="A comprehensive summary of the most significant news from the past 24 hours, already formatted with paragraph breaks using \\n\\n between paragraphs. No HTML formatting.")
    top_keywords: List[str] = Field(description="A list of 5-7 most prominent keywords or key phrases.")
    key_story_titles: List[str] = Field(description="Headlines of 3-5 key stories mentioned.")
    sentiment: str = Field(description="Overall sentiment (e.g., Positive, Negative, Neutral, Mixed).")

class LinkedSummaryOutput(BaseModel):
    summary_with_link_markers: str = Field(description="The summary text with link markers inserted. Use ==>link_start <short_name_here><== to start a link and ==>link_end<== to end it.")

# --- Simplified LLM Functions for Daily Summary Workflow ---

def generate_simple_daily_summary(articles_data: List[dict], llm_model: str = "gpt-4o-mini") -> dict:
    """Generate a daily summary that's already formatted with paragraphs using \\n\\n separators"""
    llm = ChatOpenAI(model=llm_model, temperature=0.7, max_tokens=2000)
    sys_prompt = '''
    You are an expert news editor. Your task is to create a comprehensive yet concise daily news summary 
    based on the provided articles. The summary should be well-organized into logical paragraphs, 
    separated by double newlines (\\n\\n). Focus on clarity, accuracy, and an objective tone.
    DO NOT include any HTML formatting or hyperlinks. Use only plain text with paragraph breaks.
    The output MUST be a valid JSON object.
    '''
    user_prompt = '''
    Based on the following collection of news articles from the past 24 hours, 
    please generate a daily news briefing. The 'paragraphed_summary' should be 500-700 words, 
    organized into logical paragraphs separated by \\n\\n.

    {format_instructions}

    Article Data:
    {articles_input_str}
    '''

    articles_input_parts = []
    for i, article in enumerate(articles_data):
        part = f"--- ARTICLE {i+1} ---\n"
        part += f"HEADLINE: {article.get('headline', 'N/A')}\n"
        part += f"SUMMARY: {article.get('summary_text', 'N/A')}\n\n"
        articles_input_parts.append(part)
    articles_input_str = "".join(articles_input_parts)

    parser = PydanticOutputParser(pydantic_object=SimplifiedDailySummaryOutput)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])

    res = llm.invoke(prompt_template.format_messages(articles_input_str=articles_input_str, format_instructions=parser.get_format_instructions()))
    parsed_data = parser.parse(res.content).model_dump()
    # Override date to be sure
    parsed_data['date'] = datetime.now(pytz.utc)
    return parsed_data

def insert_link_markers(summary_text: str, topics_with_short_names: List[Dict[str, str]], llm_model: str = "gpt-4o-mini") -> dict:
    """Insert link markers in the summary text where topics should be linked"""
    llm = ChatOpenAI(model=llm_model, temperature=0.2, max_tokens=2500)
    sys_prompt = '''
    You are an expert text processor. Your task is to analyze a news summary and insert link markers 
    where the text refers to specific topics from the provided list.
    
    When you identify text that clearly refers to one of the provided topics, wrap the relevant text with:
    ==>link_start <short_name><== at the beginning 
    ==>link_end<== at the end
    
    Where <short_name> is the exact short_name from the topic list.
    
    Identify the short fragments of text where the link should be inserted. Most of the
    sentences in the original text must get links.
    Every link_start marker must have a corresponding link_end marker.
    There must be some text between the link_start and link_end markers.
    Preserve all original text, spacing, and punctuation exactly.
    
    Examples:
    Given the text: In Nigeria, a devastating flooding event in Mokwa, Niger State, has 
    led to the deaths of at least 111 people, with ongoing rescue operations 
    uncovering more victims. Torrential rains and a dam collapse have exacerbated the 
    situation, displacing many residents and prompting calls for improved infrastructure 
    to prevent future disasters. This incident underscores the recurring challenges of seasonal 
    flooding affecting communities along critical waterways in the country.
    And the topic:
    Short Name: Nigeria Flood Disaster
    Title: More than 100 people killed after floods submerge market town in Nigeria
    Summary: At least 111 people have been confirmed dead in Mokwa, Niger State, Nigeria, 
    due to severe flooding caused by torrential rains and a dam collapse. The flooding has 
    displaced many residents, and ongoing rescue operations are uncovering more bodies. 
    This disaster underscores the recurring challenges of seasonal flooding in 
    Nigeria, particularly affecting communities along the Niger and Benue Rivers, 
    prompting local officials to call for improved infrastructure to mitigate future occurrences.
    You would insert the link marker here:
    At least 111 people have been ==>link_start Nigeria Flood Disaster<== confirmed dead in 
    Mokwa, Niger State, Nigeria ==>link_end<==, 
    due to severe flooding caused by torrential rains and a dam collapse. The flooding has 
    displaced many residents, and ongoing rescue operations are uncovering more bodies. 
    This disaster underscores the recurring challenges of seasonal flooding in Nigeria, 
    particularly affecting communities along the Niger and Benue Rivers, prompting local 
    officials to call for improved infrastructure to mitigate future occurrences.
    
    Given the text:
    In political news, President Trump is visiting Pittsburgh to celebrate Japan's acquisition 
    of US Steel, a deal he previously denounced as a 'disaster.' While Trump promotes the 
    acquisition as a partnership beneficial for American jobs, there is significant bipartisan 
    backlash, particularly from the United Steelworkers union, who argue it undermines national 
    security and American steelworkers. The complexities surrounding this deal highlight the 
    contentious nature of foreign investment in American industry.
    And the topic:
    Short Name: Trump Japan Steel Deal
    Title: Trump Celebrates Japan's Acquisition of US Steel amid Bipartisan Backlash
    Summary: President Donald Trump is visiting Pittsburgh to celebrate the acquisition of U.S. 
    Steel by Japan's Nippon Steel, a deal he initially opposed but now endorses as a beneficial 
    partnership for American jobs and investment. The acquisition, which has sparked bipartisan 
    criticism, particularly from President Joe Biden who blocked it on national security grounds, 
    raises concerns among labor unions about the impact on American steelworkers. Despite promises 
    of job creation and investment, the specifics of the deal remain unclear, leading to 
    skepticism regarding its implications for the U.S. manufacturing sector.
    You would insert the link marker here:
    President Donald Trump is visiting Pittsburgh to ==>link_start Trump Japan Steel Deal<== celebrate 
    the acquisition of U.S. Steel by Japan's Nippon Steel==>link_end<==, a deal he initially opposed but now 
    endorses as a beneficial partnership for American jobs and investment. The acquisition, 
    which has sparked bipartisan criticism, particularly from President Joe Biden who blocked 
    it on national security grounds, raises concerns among labor unions about the impact on 
    American steelworkers. Despite promises of job creation and investment, the specifics 
    of the deal remain unclear, leading to skepticism regarding its implications for the U.S. 
    manufacturing sector.
    '''
    
    user_prompt = '''
    Insert link markers in the following summary text. Use the format:
    ==>link_start <short_name><== (link text here) ==>link_end<==
    
    Only link text that clearly refers to one of these topics:
    
    {topics_list}
    
    Summary to process:
    {summary_text}
    
    {format_instructions}
    '''
    
    topics_list_parts = []
    for i, topic in enumerate(topics_with_short_names):
        part = f"Topic {i+1}:\n"
        part += f"  Short Name: {topic.get('short_name')}\n"
        part += f"  Title: {topic.get('title')}\n"
        part += f"  Summary: {topic.get('summary')}\n\n"
        topics_list_parts.append(part)
    topics_list = "".join(topics_list_parts) if topics_list_parts else "No topics available for linking."
    
    print("\n\n--------------------------------")
    print(summary_text)
    print("\n\n--------------------------------")
    print(topics_list)
    print("\n\n--------------------------------")

    parser = PydanticOutputParser(pydantic_object=LinkedSummaryOutput)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt)
    ])

    res = llm.invoke(prompt_template.format_messages(
        summary_text=summary_text,
        topics_list=topics_list,
        format_instructions=parser.get_format_instructions()
    ))
    parsed_data = parser.parse(res.content).model_dump()
    return parsed_data
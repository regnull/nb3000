import re
import pytz
import pprint
import html
import traceback
from datetime import datetime, timedelta
from pymongo.database import Database
from pymongo.collection import Collection

# Assuming LLM functions and Pydantic models are in cron.llm
from llm import (
    generate_initial_daily_summary, 
    generate_topic_short_name, 
    structure_plain_text_into_paragraphs, 
    segment_paragraph_for_short_name_linking, 
    DailyNewsSummary # For final object structure
)

def create_and_save_daily_summary(db: Database, stories_col: Collection, topics_col: Collection, news_summaries_col: Collection):
    """
    Orchestrates the generation of the daily news summary through a multi-step LLM process 
    and saves it to the database.
    """
    print("\nGenerating daily news summary (Workflow with Topic Short Names)...")
    one_day_ago = datetime.now(pytz.utc) - timedelta(days=1)
    
    recent_articles_cursor = stories_col.find({
        "updated": { "$gte": one_day_ago }
    }).sort("updated", -1)
    articles_for_initial_summary = list(recent_articles_cursor)
    
    if not articles_for_initial_summary:
        print("No recent articles found in the last 24 hours to generate a daily summary.")
        return

    initial_summary_input_data = []
    topic_ids_from_recent_articles = set()
    for art in articles_for_initial_summary:
        initial_summary_input_data.append({
            "headline": art.get('headline', 'N/A'),
            "summary_text": art.get('summary', {}).get('summary', 'N/A') if isinstance(art.get('summary'), dict) else str(art.get('summary', 'N/A')),
        })
        if art.get('topic'): # ObjectId
            topic_ids_from_recent_articles.add(art.get('topic'))

    try:
        # --- Step 1: Generate initial plain text summary and metadata --- (LLM 1)
        print("Step 1: Generating initial plain text summary...")
        initial_summary_output = generate_initial_daily_summary(initial_summary_input_data)
        plain_text_summary = initial_summary_output.get('plain_text_summary')
        if not plain_text_summary:
            print("Error: LLM (Step 1) did not return a plain text summary.")
            return # Changed from exit() to return for better modularity

        # --- Step 2a: Generate Short Names for relevant topics --- (LLM 2a for each topic)
        print("Step 2a: Generating short names for relevant topics...")
        enriched_linkable_topics = []
        short_name_to_topic_id_map = {}
        if topic_ids_from_recent_articles:
            fetched_topics = list(topics_col.find({"_id": {"$in": list(topic_ids_from_recent_articles)}}))
            for topic_doc in fetched_topics:
                topic_id_str = str(topic_doc.get('_id'))
                topic_title = topic_doc.get('summary', {}).get('title', 'N/A')
                topic_s_summary = topic_doc.get('summary', {}).get('summary', 'N/A')
                
                existing_short_name = topic_doc.get('short_name') 
                short_name_to_use = None

                if existing_short_name:
                    short_name_to_use = existing_short_name
                    print(f"  Using existing short name '{short_name_to_use}' for topic ID {topic_id_str}")
                else:
                    try:
                        print(f"  Generating new short name for topic ID {topic_id_str}...")
                        short_name_output = generate_topic_short_name(topic_title, topic_s_summary)
                        generated_short_name = short_name_output.get('short_name')
                        if generated_short_name:
                            short_name_to_use = generated_short_name
                            topics_col.update_one(
                                {'_id': topic_doc.get('_id')},
                                {'$set': {'short_name': short_name_to_use}}
                            )
                            print(f"    Generated and saved short name '{short_name_to_use}' for topic ID {topic_id_str}")
                        else:
                            print(f"  Warning: Could not generate short name for topic ID {topic_id_str}. It will not be linkable by short name.")
                    except Exception as e_sn:
                        print(f"  Error generating short name for topic ID {topic_id_str}: {e_sn}")

                if short_name_to_use:
                    enriched_linkable_topics.append({
                        "topic_id": topic_id_str,
                        "title": topic_title,
                        "summary": topic_s_summary,
                        "short_name": short_name_to_use
                    })
                    short_name_to_topic_id_map[short_name_to_use] = topic_id_str 
        
        if not enriched_linkable_topics:
             print("No linkable topics with short names generated. Links will not be added (or fewer will be).")

        # --- Step 2b: Structure plain text summary into paragraphs --- (LLM 2b)
        print("Step 2b: Structuring plain text summary into paragraphs...")
        paragraphed_text_output = structure_plain_text_into_paragraphs(plain_text_summary)
        plain_text_paragraphs = paragraphed_text_output.get('paragraphs', [])
        if not plain_text_paragraphs:
            print("Error: LLM (Step 2b) did not return paragraphs. Using original plain summary as one paragraph.")
            plain_text_paragraphs = [plain_text_summary] 

        # --- Step 3 & 4: Segment each paragraph for (short_name) links and construct final HTML ---
        print("Step 3 & 4: Segmenting paragraphs for short_name links and constructing HTML...")
        all_html_paragraphs = []
        
        for para_text in plain_text_paragraphs:
            if not para_text.strip(): continue

            segmented_para_output = segment_paragraph_for_short_name_linking(para_text, enriched_linkable_topics)
            para_chunks = segmented_para_output.get('chunks', [])

            if not para_chunks:
                print(f"Warning: Link segmentation (Step 3) failed for paragraph. Using plain text: '{para_text[:50]}...'")
                all_html_paragraphs.append(f"<p>{html.escape(para_text)}</p>")
                continue

            current_paragraph_html_parts = []
            previous_chunk_ended_sentence = True 

            for i_chunk, chunk_data in enumerate(para_chunks):
                current_text_segment = ""
                is_potential_link = False
                identified_short_name = None

                if chunk_data.get('type') == 'potential_link':
                    current_text_segment = chunk_data.get('link_text', 'details')
                    identified_short_name = chunk_data.get("identified_short_name")
                    is_potential_link = True
                elif chunk_data.get('type') == 'text':
                    current_text_segment = chunk_data.get('content', '')
                else: 
                    current_text_segment = str(chunk_data)
                
                if not current_text_segment.strip() and not is_potential_link:
                    continue 

                if previous_chunk_ended_sentence and current_text_segment and current_text_segment[0].islower():
                    current_text_segment = current_text_segment[0].upper() + current_text_segment[1:]
                
                html_segment_to_add = ""
                if is_potential_link and identified_short_name:
                    actual_topic_id = short_name_to_topic_id_map.get(identified_short_name)
                    if actual_topic_id:
                        link_text_escaped = html.escape(current_text_segment) 
                        html_segment_to_add = f'<a href="/topic/{html.escape(actual_topic_id)}">{link_text_escaped}</a>'
                    else:
                        print(f"Warning: LLM identified short_name '{identified_short_name}' but no matching topic_id found. Treating as text.")
                        html_segment_to_add = html.escape(current_text_segment)
                else:
                    html_segment_to_add = html.escape(current_text_segment)

                # Smart spacing: Add a space BEFORE the current segment if needed
                if current_paragraph_html_parts and html_segment_to_add: 
                    # Check if last part doesn't end with space/newline/tag-end AND current doesn't start with space/newline/tag-start
                    if not current_paragraph_html_parts[-1].endswith((' ', '\n', '>')) and \
                       not html_segment_to_add.startswith((' ', '\n', '<')):
                         current_paragraph_html_parts.append(' ')
                
                current_paragraph_html_parts.append(html_segment_to_add)

                # Determine if this segment ended a sentence (based on its original text content)
                current_segment_ended_sentence = False
                if current_text_segment.strip(): # Check original text, not the HTML part
                    last_char = current_text_segment.strip()[-1]
                    if last_char in ['.', '!', '?']:
                        current_segment_ended_sentence = True
                
                previous_chunk_ended_sentence = current_segment_ended_sentence

                # Add a space AFTER this segment if it ended a sentence AND it's not the last chunk in the paragraph
                if current_segment_ended_sentence and (i_chunk < len(para_chunks) - 1):
                    # Ensure the very next chunk isn't just punctuation that should attach directly
                    # This requires looking ahead to para_chunks[i_chunk+1]
                    # For simplicity now, just add the space. Regex cleanup will handle multiple spaces.
                    current_paragraph_html_parts.append(' ')
            
            inner_html = "".join(current_paragraph_html_parts)
            all_html_paragraphs.append(f"<p>{inner_html.strip()}</p>") # .strip() inner_html

        final_html_summary = "\n".join(all_html_paragraphs)
        final_html_summary = re.sub(r' + ', ' ', final_html_summary).strip()
        final_html_summary = re.sub(r'<p>\s*</p>', '', final_html_summary, flags=re.IGNORECASE)

        # --- Step 5: Assemble final DailyNewsSummary object and save to DB ---
        print("Step 5: Saving final daily summary to MongoDB...")
        final_daily_summary_doc = DailyNewsSummary(
            date=initial_summary_output.get('date', datetime.now(pytz.utc)), 
            title=initial_summary_output.get('title', 'Daily News Summary'),
            overall_summary=final_html_summary, 
            top_keywords=initial_summary_output.get('top_keywords', []),
            key_story_titles=initial_summary_output.get('key_story_titles', []),
            sentiment=initial_summary_output.get('sentiment', 'Neutral')
        ).model_dump()

        final_daily_summary_doc['date'] = datetime.now(pytz.utc)

        news_summaries_col.insert_one(final_daily_summary_doc)
        print("Successfully generated and saved HTML daily news summary with paragraphs.")
        pprint.pprint({
            "title": final_daily_summary_doc['title'],
            "date": final_daily_summary_doc['date'],
            "html_summary_snippet": final_html_summary[:300] + "..." 
        })

    except Exception as e:
        print(f"An error occurred during the new 3-step daily summary workflow: {e}")
        traceback.print_exc() 
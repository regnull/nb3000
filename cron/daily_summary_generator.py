import re
import pytz
import pprint
import html
import traceback
from datetime import datetime, timedelta
from pymongo.database import Database
from pymongo.collection import Collection

# Updated imports for simplified approach
from llm import (
    generate_simple_daily_summary, 
    insert_link_markers,
    generate_topic_short_name,
    DailyNewsSummary
)

def create_and_save_daily_summary(db: Database, stories_col: Collection, topics_col: Collection, news_summaries_col: Collection):
    """
    Simplified daily news summary generation with 3 steps:
    1. Generate text summary with paragraphs
    2. Insert link markers 
    3. Replace markers with HTML links
    """
    print("\nGenerating daily news summary (Simplified 3-step approach)...")
    one_day_ago = datetime.now(pytz.utc) - timedelta(days=1)
    
    recent_articles_cursor = stories_col.find({
        "updated": { "$gte": one_day_ago }
    }).sort("updated", -1)
    articles_for_summary = list(recent_articles_cursor)
    
    if not articles_for_summary:
        print("No recent articles found in the last 24 hours to generate a daily summary.")
        return

    # Prepare input data for summary generation
    summary_input_data = []
    topic_ids_from_recent_articles = set()
    for art in articles_for_summary:
        summary_input_data.append({
            "headline": art.get('headline', 'N/A'),
            "summary_text": art.get('summary', {}).get('summary', 'N/A') if isinstance(art.get('summary'), dict) else str(art.get('summary', 'N/A')),
        })
        if art.get('topic'):
            topic_ids_from_recent_articles.add(art.get('topic'))

    try:
        # --- Step 1: Generate initial summary with paragraphs ---
        print("Step 1: Generating summary with paragraphs...")
        summary_output = generate_simple_daily_summary(summary_input_data)
        paragraphed_summary = summary_output.get('paragraphed_summary')
        if not paragraphed_summary:
            print("Error: Failed to generate paragraphed summary.")
            return
        
        print("\n\n--------------------------------")
        print(paragraphed_summary)

        # --- Step 2: Prepare topics with short names for linking ---
        print("Step 2: Preparing topics with short names...")
        topics_with_short_names = []
        short_name_to_topic_id_map = {}
        
        if topic_ids_from_recent_articles:
            fetched_topics = list(topics_col.find({"_id": {"$in": list(topic_ids_from_recent_articles)}}))
            for topic_doc in fetched_topics:
                topic_id_str = str(topic_doc.get('_id'))
                topic_title = topic_doc.get('summary', {}).get('title', 'N/A')
                topic_summary = topic_doc.get('summary', {}).get('summary', 'N/A')
                
                # Get or generate short name
                existing_short_name = topic_doc.get('short_name') 
                short_name_to_use = None

                if existing_short_name:
                    short_name_to_use = existing_short_name
                    print(f"  Using existing short name '{short_name_to_use}' for topic ID {topic_id_str}")
                else:
                    try:
                        print(f"  Generating new short name for topic ID {topic_id_str}...")
                        short_name_output = generate_topic_short_name(topic_title, topic_summary)
                        generated_short_name = short_name_output.get('short_name')
                        if generated_short_name:
                            short_name_to_use = generated_short_name
                            topics_col.update_one(
                                {'_id': topic_doc.get('_id')},
                                {'$set': {'short_name': short_name_to_use}}
                            )
                            print(f"    Generated and saved short name '{short_name_to_use}' for topic ID {topic_id_str}")
                        else:
                            print(f"  Warning: Could not generate short name for topic ID {topic_id_str}.")
                    except Exception as e:
                        print(f"  Error generating short name for topic ID {topic_id_str}: {e}")

                if short_name_to_use:
                    topics_with_short_names.append({
                        "topic_id": topic_id_str,
                        "title": topic_title,
                        "summary": topic_summary,
                        "short_name": short_name_to_use
                    })
                    short_name_to_topic_id_map[short_name_to_use] = topic_id_str

        # --- Step 3: Insert link markers ---
        print("Step 3: Inserting link markers...")
        if topics_with_short_names:
            linked_output = insert_link_markers(paragraphed_summary, topics_with_short_names)
            summary_with_markers = linked_output.get('summary_with_link_markers', paragraphed_summary)
        else:
            print("No topics with short names available for linking.")
            summary_with_markers = paragraphed_summary
        print("\n\n--------------------------------")
        print(summary_with_markers)
        print("\n\n--------------------------------")

        # --- Step 4: Replace markers with HTML links ---
        print("Step 4: Converting markers to HTML links...")
        final_html_summary = convert_markers_to_html(summary_with_markers, short_name_to_topic_id_map)
        print("\n\n--------------------------------")
        print(final_html_summary)
        print("\n\n--------------------------------")

        # --- Step 5: Save to database ---
        print("Step 5: Saving final daily summary to MongoDB...")
        final_daily_summary_doc = DailyNewsSummary(
            date=summary_output.get('date', datetime.now(pytz.utc)), 
            title=summary_output.get('title', 'Daily News Summary'),
            overall_summary=final_html_summary, 
            top_keywords=summary_output.get('top_keywords', []),
            key_story_titles=summary_output.get('key_story_titles', []),
            sentiment=summary_output.get('sentiment', 'Neutral')
        ).model_dump()

        final_daily_summary_doc['date'] = datetime.now(pytz.utc)

        news_summaries_col.insert_one(final_daily_summary_doc)
        print("Successfully generated and saved simplified daily news summary.")
        pprint.pprint({
            "title": final_daily_summary_doc['title'],
            "date": final_daily_summary_doc['date'],
            "html_summary_snippet": final_html_summary[:300] + "..." 
        })

    except Exception as e:
        print(f"An error occurred during the simplified daily summary workflow: {e}")
        traceback.print_exc()

def convert_markers_to_html(summary_with_markers: str, short_name_to_topic_id_map: dict) -> str:
    """
    Convert link markers to actual HTML links and format paragraphs
    
    Converts:
    ==>link_start <short_name><== (link text) ==>link_end<==
    
    To:
    <a href="/topic/{topic_id}">(link text)</a>
    """
    print("Converting link markers to HTML...")
    
    # Split into paragraphs first
    paragraphs = summary_with_markers.split('\n\n')
    html_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Process link markers in this paragraph
        processed_paragraph = paragraph
        
        # Find all link markers using regex
        # Pattern: ==>link_start <short_name><== ... ==>link_end<==
        link_pattern = r'==>link_start\s+([^<]+)<==\s*(.*?)\s*==>link_end<=='
        
        def replace_link(match):
            short_name = match.group(1).strip()
            link_text = match.group(2).strip()
            
            topic_id = short_name_to_topic_id_map.get(short_name)
            if topic_id:
                # Escape the link text for HTML
                escaped_link_text = html.escape(link_text)
                # Escape the topic ID for the URL
                escaped_topic_id = html.escape(topic_id)
                # Create the HTML link with proper escaping
                return f'<a href="/topic/{escaped_topic_id}">{escaped_link_text}</a>'
            else:
                print(f"Warning: No topic ID found for short name '{short_name}'. Using plain text.")
                return html.escape(link_text)
        
        # Replace all link markers in the paragraph
        processed_paragraph = re.sub(link_pattern, replace_link, processed_paragraph, flags=re.DOTALL)
        
        # Escape any remaining text that's not in HTML tags
        # First, temporarily replace our links with placeholders
        link_placeholders = []
        def save_link(match):
            link_placeholders.append(match.group(0))
            return f"__LINK_PLACEHOLDER_{len(link_placeholders)-1}__"
        
        # Save all links and replace with placeholders
        processed_paragraph = re.sub(r'<a[^>]*>.*?</a>', save_link, processed_paragraph, flags=re.DOTALL)
        
        # Escape the remaining text
        escaped_paragraph = html.escape(processed_paragraph)
        
        # Restore the links
        for i, placeholder in enumerate(link_placeholders):
            escaped_paragraph = escaped_paragraph.replace(f"__LINK_PLACEHOLDER_{i}__", placeholder)
        
        html_paragraphs.append(f"<p>{escaped_paragraph}</p>")
    
    final_html = "\n".join(html_paragraphs)
    
    # Clean up any multiple spaces
    final_html = re.sub(r' +', ' ', final_html)
    
    # Remove any empty paragraphs
    final_html = re.sub(r'<p>\s*</p>', '', final_html, flags=re.IGNORECASE)
    
    print(f"Converted {len(html_paragraphs)} paragraphs to HTML.")
    return final_html.strip() 
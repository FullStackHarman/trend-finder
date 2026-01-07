import streamlit as st
import pandas as pd
import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import urllib.parse

# --- CONFIGURATION ---
st.set_page_config(page_title="Viral Content Engine (With Scraper)", layout="wide")

# --- FUNCTIONS ---

def scrape_article(url):
    """
    Fetches the article URL and extracts the main text to prevent hallucinations.
    """
    # Browser masquerade to prevent being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # requests.get automatically follows the Google News redirect
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Heuristic: Most articles have content in <p> tags. 
        # We grab all paragraph text and join it.
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text() for p in paragraphs])
        
        # Clean up whitespace
        text_content = " ".join(text_content.split())
        
        # Limit content to 4000 chars to save tokens and fit context window
        if len(text_content) > 4000:
            return text_content[:4000] + "...(truncated)"
        return text_content
        
    except Exception as e:
        print(f"Scraping failed: {e}")
        return None

def analyze_virality(topics, api_key, platform):
    """ Step 1: Analyze Trends (Headlines Only) """
    client = OpenAI(api_key=api_key)
    # We store the 'link' in the analysis so we can use it later
    topics_data = [{"title": t['title'], "link": t['link']} for t in topics]
    
    # Create a simplified list for the prompt
    topics_str = "\n".join([f"{i+1}. {t['title']}" for i, t in enumerate(topics_data[:8])])

    prompt = f"""
    You are a viral content strategist for {platform}. Analyze these stories:
    {topics_str}
    
    For each story, provide:
    1. 'score': Viral Potential (0-100).
    2. 'headline': A catchy {platform} style hook.
    3. 'reason': Why it works.
    4. 'visual_prompt': A detailed description for an AI image generator.
    
    Return ONLY a raw list of dictionaries (JSON) with keys: "topic", "score", "headline", "reason", "visual_prompt".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "system", "content": "You are a JSON machine."}, {"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"): content = content.split("```")[1].strip()
        if content.startswith("json"): content = content[4:].strip()
        
        results = json.loads(content)
        
        # CRITICAL: Re-attach the original links to the results
        # The AI returns a list in the same order. We merge the link back in.
        for i, res in enumerate(results):
            if i < len(topics_data):
                res['link'] = topics_data[i]['link']
                
        return results
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        return []

def generate_post(topic_dict, platform, api_key, article_context=None):
    """ Step 2: Write Text (Now with Source Context) """
    client = OpenAI(api_key=api_key)
    
    context_instruction = ""
    if article_context:
        context_instruction = f"""
        REAL ARTICLE DATA:
        "{article_context}"
        
        INSTRUCTIONS:
        Use the data above to write the post. Cite specific facts/figures from the text. 
        Do not hallucinate info not present in the source.
        """
    else:
        context_instruction = "No source text available. Write creatively based on the headline."

    prompt = f"""
    Platform: {platform}
    Topic: {topic_dict['topic']}
    Hook: {topic_dict['headline']}
    
    {context_instruction}
    
    Output Format:
    Write a high-engagement post optimized for {platform}.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def generate_image(prompt, api_key):
    """ Step 3: Create Image (DALL-E 3) """
    client = OpenAI(api_key=api_key)
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        st.error(f"Image Error: {e}")
        return None

def get_news_feed(query, country_code, language_code):
    """ Fetch RSS Feed """
    base_url = "https://news.google.com/rss"
    if query:
        encoded = urllib.parse.quote(query)
        params = f"/search?q={encoded}&hl={language_code}&gl={country_code}&ceid={country_code}:{language_code.split('-')[0]}"
    else:
        params = f"?hl={language_code}&gl={country_code}&ceid={country_code}:{language_code.split('-')[0]}"
    
    try:
        feed = feedparser.parse(base_url + params)
        # We grab the link too!
        return [{"title": x.title, "link": x.link} for x in feed.entries[:10]]
    except Exception:
        return []

# --- APP UI ---
st.title("🤖 Fact-Checked Content Engine")
st.caption("Scrapes real article content to minimize AI hallucinations.")

with st.sidebar:
    st.header("Settings")
    openai_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    search_query = st.text_input("Search Niche", placeholder="e.g. Spices, UPSC, Tech")
    target_platform = st.selectbox("Platform", ["LinkedIn", "Instagram", "Twitter/X", "Blog"])
    
    if 'analyzed_data' not in st.session_state: st.session_state.analyzed_data = None
    if 'generated_post' not in st.session_state: st.session_state.generated_post = None
    if 'generated_image' not in st.session_state: st.session_state.generated_image = None
    if 'source_url' not in st.session_state: st.session_state.source_url = None

    if st.button("1. Find Trends"):
        if openai_key:
            with st.spinner("Scanning News..."):
                raw_news = get_news_feed(search_query, "IN", "en-IN")
                if raw_news:
                    results = analyze_virality(raw_news, openai_key, target_platform)
                    if results:
                        st.session_state.analyzed_data = pd.DataFrame(results)
                        st.success("Trends Found!")

# --- DISPLAY ---
if st.session_state.analyzed_data is not None:
    df = st.session_state.analyzed_data.sort_values(by="score", ascending=False)
    
    st.subheader("1. Trending Topics")
    st.dataframe(df[['score', 'headline']].style.background_gradient(subset=['score'], cmap="Greens"), use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("2. Select & Generate")
        options = [f"({r['score']}) {r['headline']}" for i, r in df.iterrows()]
        selected_str = st.selectbox("Choose Topic:", options)
        selected_idx = options.index(selected_str)
        selected_data = df.iloc[selected_idx]
        
        st.caption(f"**Strategy:** {selected_data['reason']}")
        st.caption(f"**Source Link:** {selected_data.get('link', 'No link')}")
        
        if st.button("Generate Post & Image 🚀"):
            if openai_key:
                # 1. Scrape Content
                article_text = None
                with st.spinner("Reading article source..."):
                    if 'link' in selected_data:
                        article_text = scrape_article(selected_data['link'])
                        if article_text:
                            st.toast("Article scraped successfully!", icon="✅")
                        else:
                            st.toast("Could not scrape article. Using headline only.", icon="⚠️")
                
                # 2. Generate Text
                with st.spinner("Writing factual content..."):
                    text = generate_post(selected_data, target_platform, openai_key, article_text)
                    st.session_state.generated_post = text
                    st.session_state.source_url = selected_data.get('link')
                    
                # 3. Generate Image
                with st.spinner("Creating visual..."):
                    img_url = generate_image(selected_data['visual_prompt'], openai_key)
                    st.session_state.generated_image = img_url
                    st.success("Done!")

    with col2:
        st.subheader("3. Final Result")
        if st.session_state.generated_image:
            st.image(st.session_state.generated_image, caption="AI Generated Visual")
            
        if st.session_state.generated_post:
            st.text_area("Copy Text:", value=st.session_state.generated_post, height=400)
            if st.session_state.source_url:
                st.markdown(f"[Read Original Source]({st.session_state.source_url})")

else:
    st.info("👈 Enter API Key and Search to begin.")
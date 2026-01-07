import streamlit as st
import pandas as pd
import feedparser
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import urllib.parse

# --- CONFIGURATION ---
st.set_page_config(page_title="Multi-Source Trend Hunter", layout="wide")

# --- DATA FETCHING FUNCTIONS ---

def get_google_news(query, country_code="IN", language_code="en-IN"):
    """ Source 1: Google News RSS """
    base_url = "https://news.google.com/rss"
    if query:
        encoded = urllib.parse.quote(query)
        params = f"/search?q={encoded}&hl={language_code}&gl={country_code}&ceid={country_code}:{language_code.split('-')[0]}"
    else:
        params = f"?hl={language_code}&gl={country_code}&ceid={country_code}:{language_code.split('-')[0]}"
    
    try:
        feed = feedparser.parse(base_url + params)
        # Standardize keys: title, link, source
        return [{"title": x.title, "link": x.link, "source": "Google News"} for x in feed.entries[:10]]
    except Exception:
        return []

def get_reddit(query):
    """ Source 2: Reddit JSON API """
    if query:
        url = f"https://www.reddit.com/search.json?q={urllib.parse.quote(query)}&sort=relevance&limit=10"
    else:
        url = "https://www.reddit.com/r/popular.json?limit=10"

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        posts = []
        for item in data['data']['children']:
            post = item['data']
            if not post.get('over_18'):
                posts.append({
                    "title": post['title'],
                    "link": f"https://www.reddit.com{post['permalink']}",
                    "source": f"Reddit (r/{post['subreddit']})"
                })
        return posts
    except Exception as e:
        st.error(f"Reddit Error: {e}")
        return []

def get_hacker_news(query):
    """ Source 3: Hacker News """
    if query:
        url = f"https://hnrss.org/newest?q={urllib.parse.quote(query)}"
    else:
        url = "https://hnrss.org/frontpage"

    try:
        feed = feedparser.parse(url)
        return [{"title": x.title, "link": x.link, "source": "Hacker News"} for x in feed.entries[:10]]
    except Exception:
        return []

def scrape_article(url):
    """ Helper: Extracts text from the link for context """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        text = " ".join(text.split())
        if len(text) > 4000: return text[:4000] + "..."
        return text
    except Exception:
        return None

# --- AI FUNCTIONS ---

def analyze_virality(topics, api_key, platform):
    client = OpenAI(api_key=api_key)
    topics_data = [{"title": t['title'], "link": t['link'], "source": t['source']} for t in topics]
    
    # Prompt engineering
    topics_str = "\n".join([f"{i+1}. [{t['source']}] {t['title']}" for i, t in enumerate(topics_data[:8])])

    prompt = f"""
    Act as a viral content strategist for {platform}. Analyze these trending items:
    {topics_str}
    
    For each item, return a JSON object with:
    1. 'score': Viral Potential (0-100).
    2. 'headline': A catchy {platform} hook.
    3. 'reason': Why it works.
    4. 'visual_prompt': Image generation prompt.
    
    Output strictly a JSON list.
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
        
        # --- THE FIX IS HERE ---
        # We manually attach the original title as 'topic' so generate_post can find it.
        for i, res in enumerate(results):
            if i < len(topics_data):
                res['link'] = topics_data[i]['link']
                res['source'] = topics_data[i]['source']
                res['topic'] = topics_data[i]['title'] # <--- Added this line
                
        return results
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        return []

def generate_post(topic_dict, platform, api_key, context=None):
    client = OpenAI(api_key=api_key)
    context_str = f"Source Text: {context}" if context else "No source text available."
    
    # We use .get() to avoid crashing if 'topic' is missing, but our fix above ensures it is there.
    topic_title = topic_dict.get('topic', topic_dict.get('headline', 'Unknown Topic'))

    prompt = f"""
    Platform: {platform}
    Topic: {topic_title}
    Hook: {topic_dict.get('headline', '')}
    {context_str}
    
    Write a high-engagement, viral post. 
    If source text is provided, include specific facts from it.
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
    client = OpenAI(api_key=api_key)
    try:
        response = client.images.generate(
            model="dall-e-3", prompt=prompt, size="1024x1024", quality="standard", n=1
        )
        return response.data[0].url
    except Exception:
        return None

# --- APP UI ---
st.title("🌎 Multi-Source Trend Hunter")
st.caption("Scrape Google, Reddit, and Hacker News for viral opportunities.")

with st.sidebar:
    st.header("Settings")
    openai_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    # Source Selector
    data_source = st.selectbox("Select Data Source", ["Google News", "Reddit", "Hacker News"])
    
    search_query = st.text_input("Search Keyword (Optional)", placeholder="e.g. AI, Spices, Politics")
    target_platform = st.selectbox("Target Platform", ["LinkedIn", "Twitter/X", "Instagram", "Blog"])
    
    if st.button("Find Trends"):
        if openai_key:
            with st.spinner(f"Scraping {data_source}..."):
                raw_data = []
                
                # Route request
                if data_source == "Google News":
                    raw_data = get_google_news(search_query)
                elif data_source == "Reddit":
                    raw_data = get_reddit(search_query)
                elif data_source == "Hacker News":
                    raw_data = get_hacker_news(search_query)
                
                if raw_data:
                    results = analyze_virality(raw_data, openai_key, target_platform)
                    if results:
                        st.session_state.analyzed_data = pd.DataFrame(results)
                        st.success(f"Found {len(results)} trends from {data_source}!")
                else:
                    st.warning("No data found. Try a different source or keyword.")

# --- DISPLAY ---
if 'analyzed_data' in st.session_state and st.session_state.analyzed_data is not None:
    df = st.session_state.analyzed_data.sort_values(by="score", ascending=False)
    
    st.subheader("1. Trending Topics")
    st.dataframe(
        df[['score', 'source', 'headline']].style.background_gradient(subset=['score'], cmap="Greens"), 
        use_container_width=True
    )
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("2. Select & Generate")
        # Creating options list
        options = [f"({r['score']}) {r['headline']}" for i, r in df.iterrows()]
        selected_str = st.selectbox("Choose Topic:", options)
        
        # Get selected row safely
        selected_idx = options.index(selected_str)
        selected_data = df.iloc[selected_idx]
        
        st.info(f"**Why it works:** {selected_data['reason']}")
        st.caption(f"**Original Source:** {selected_data['source']}")
        
        if st.button("Generate Post & Image 🚀"):
            if openai_key:
                # Scrape
                with st.spinner("Reading source content..."):
                    context = scrape_article(selected_data['link'])
                
                # Generate Text
                with st.spinner("Writing post..."):
                    text = generate_post(selected_data, target_platform, openai_key, context)
                    st.session_state.generated_post = text
                
                # Generate Image
                with st.spinner("Creating image..."):
                    img = generate_image(selected_data['visual_prompt'], openai_key)
                    st.session_state.generated_image = img
                    st.success("Done!")

    with col2:
        st.subheader("3. Final Result")
        if 'generated_image' in st.session_state and st.session_state.generated_image:
            st.image(st.session_state.generated_image)
        if 'generated_post' in st.session_state and st.session_state.generated_post:
            st.text_area("Content", value=st.session_state.generated_post, height=400)
            if 'link' in selected_data:
                 st.markdown(f"[View Original]({selected_data['link']})")
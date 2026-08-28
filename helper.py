import os
import re
from collections import Counter
from urllib.parse import urlparse
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    sia = SentimentIntensityAnalyzer()
except Exception:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

extract = URLExtract()

def get_stop_words():
    stop_words = {
        'the', 'is', 'a', 'to', 'and', 'in', 'of', 'for', 'it', 'on', 'with',
        'this', 'that', 'you', 'i', 'me', 'my', 'we', 'our', 'he', 'she', 'they',
        'hai', 'ye', 'wo', 'ko', 'ki', 'ke', 'ka', 'se', 'me', 'mai', 'aur', 'kya',
        'ho', 'tha', 'thi', 'the', 'bhi', 'kuch', 'nahi', 'raha', 'rahi', 'rahe',
        'par', 'pe', 'toh', 'ab', 'jab', 'tab', 'jo', 'kar', 'karo', 'kare', 'diya',
        'de', 'hoga', 'hogi', 'hoge', 'hua', 'hui', 'hue', 'apne', 'mera', 'meri',
        'mere', 'hum', 'tum', 'aap', 'uske', 'iski', 'iske', 'unke', 'sab', 'sirf',
        'media', 'omitted', 'deleted', 'message', 'this', 'was', 'image', 'video',
        'audio', 'sticker', 'attached', 'document'
    }
    stop_file = os.path.join(os.path.dirname(__file__), 'stop_hinglish.txt')
    if os.path.exists(stop_file):
        try:
            with open(stop_file, 'r', encoding='utf-8') as f:
                stop_words.update(f.read().split())
        except Exception:
            pass
    return stop_words


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(str(message).split())

    media_pattern = r'<Media omitted>|This message was deleted|image omitted|video omitted|audio omitted|sticker omitted'
    num_media_messages = df[df['message'].astype(str).str.contains(media_pattern, case=False, regex=True, na=False)].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(str(message)))

    num_emojis = sum(len([c for c in str(msg) if emoji.is_emoji(c)]) for msg in df['message'])
    avg_words = round(len(words) / num_messages, 1) if num_messages > 0 else 0
    active_days = df['only_date'].nunique() if 'only_date' in df else 0

    return {
        'num_messages': num_messages,
        'num_words': len(words),
        'num_media': num_media_messages,
        'num_links': len(links),
        'num_emojis': num_emojis,
        'avg_words_per_msg': avg_words,
        'active_days': active_days
    }


def most_busy_users(df):
    temp = df[df['user'] != 'group_notification']
    counts = temp['user'].value_counts()
    x = counts.head(10)
    percent_df = round((counts / temp.shape[0]) * 100, 2).reset_index()
    percent_df.columns = ['User', 'Percentage (%)']
    percent_df['Messages'] = counts.values
    return x, percent_df


def create_wordcloud(selected_user, df):
    stop_words = get_stop_words()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification'].copy()
    media_pattern = r'<Media omitted>|This message was deleted|image omitted|video omitted|audio omitted|sticker omitted'
    temp = temp[~temp['message'].astype(str).str.contains(media_pattern, case=False, regex=True, na=False)]

    def clean_text(message):
        cleaned = []
        for word in str(message).lower().split():
            clean_word = ''.join(e for e in word if e.isalnum())
            if clean_word and clean_word not in stop_words and len(clean_word) > 2 and not clean_word.isnumeric():
                cleaned.append(clean_word)
        return " ".join(cleaned)

    temp['cleaned_message'] = temp['message'].apply(clean_text)
    text = temp['cleaned_message'].astype(str).str.cat(sep=" ").strip()

    if not text:
        return None

    wc = WordCloud(
        width=800,
        height=400,
        min_font_size=10,
        background_color='#0e1117',
        colormap='viridis',
        collocations=False
    )
    return wc.generate(text)


def most_common_words(selected_user, df):
    stop_words = get_stop_words()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    media_pattern = r'<Media omitted>|This message was deleted|image omitted|video omitted|audio omitted|sticker omitted'
    temp = temp[~temp['message'].astype(str).str.contains(media_pattern, case=False, regex=True, na=False)]

    words = []
    for message in temp['message'].astype(str):
        for word in message.lower().split():
            clean_word = ''.join(e for e in word if e.isalnum())
            if clean_word and clean_word not in stop_words and len(clean_word) > 2 and not clean_word.isnumeric():
                words.append(clean_word)

    if not words:
        return pd.DataFrame(columns=['word', 'count'])
    
    return pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in str(message) if emoji.is_emoji(c)])

    emoji_counts = Counter(emojis)
    if not emoji_counts:
        return pd.DataFrame(columns=['emoji', 'count', 'percentage'])

    emoji_df = pd.DataFrame(emoji_counts.most_common(len(emoji_counts)), columns=['emoji', 'count'])
    total_emojis = emoji_df['count'].sum()
    emoji_df['percentage'] = round((emoji_df['count'] / total_emojis) * 100, 2)
    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(f"{timeline['month'][i]} {timeline['year'][i]}")
    timeline['time'] = time
    timeline.sort_values(by=['year', 'month_num'], inplace=True)
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily = df.groupby('only_date').count()['message'].reset_index()
    daily.rename(columns={'only_date': 'date', 'message': 'messages'}, inplace=True)
    daily.sort_values('date', inplace=True)
    return daily


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = df['day_name'].value_counts().reindex(days_order).fillna(0).astype(int)
    return counts


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    counts = df['month'].value_counts().reindex([m for m in months_order if m in df['month'].values]).fillna(0).astype(int)
    return counts


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    user_heatmap = user_heatmap.reindex([d for d in days_order if d in user_heatmap.index])
    return user_heatmap


# ==========================================
# 🚀 NEW FEATURES
# ==========================================

# 1. SENTIMENT ANALYSIS (VADER)
def analyze_sentiment(selected_user, df):
    temp = df[df['user'] != 'group_notification'].copy()
    if selected_user != 'Overall':
        temp = temp[temp['user'] == selected_user]

    media_pattern = r'<Media omitted>|This message was deleted|image omitted|video omitted|audio omitted|sticker omitted'
    temp = temp[~temp['message'].astype(str).str.contains(media_pattern, case=False, regex=True, na=False)]

    if temp.empty:
        return None

    def get_vader_score(text):
        return sia.polarity_scores(str(text))['compound']

    temp['compound'] = temp['message'].apply(get_vader_score)

    def categorize_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    temp['sentiment'] = temp['compound'].apply(categorize_sentiment)
    sentiment_counts = temp['sentiment'].value_counts()
    
    # Sentiment timeline by month
    sentiment_monthly = temp.groupby(['year', 'month_num', 'sentiment']).size().unstack(fill_value=0).reset_index()
    
    # User-level sentiment breakdown
    user_sentiment = None
    if selected_user == 'Overall':
        user_sentiment = temp.groupby('user')['compound'].mean().reset_index()
        user_sentiment.columns = ['User', 'Avg Sentiment Score']
        user_sentiment['Status'] = user_sentiment['Avg Sentiment Score'].apply(
            lambda x: '😊 Very Positive' if x > 0.15 else ('😃 Positive' if x > 0.03 else ('😐 Neutral' if x >= -0.03 else '😠 Negative'))
        )
        user_sentiment.sort_values(by='Avg Sentiment Score', ascending=False, inplace=True)

    return {
        'counts': sentiment_counts,
        'avg_compound': round(temp['compound'].mean(), 3),
        'total_analyzed': len(temp),
        'user_sentiment': user_sentiment,
        'sentiment_df': temp
    }


# 2. RESPONSE TIME & INTERACTION DYNAMICS
def response_time_analysis(df):
    temp = df[df['user'] != 'group_notification'].copy()
    temp.sort_values('date', inplace=True)
    temp['prev_user'] = temp['user'].shift(1)
    temp['prev_date'] = temp['date'].shift(1)

    # Only look at turns where User B replies to User A (different user)
    reply_df = temp[(temp['user'] != temp['prev_user']) & (temp['prev_user'].notnull())].copy()
    reply_df['diff_minutes'] = (reply_df['date'] - reply_df['prev_date']).dt.total_seconds() / 60.0

    # Filter realistic replies (between 5 seconds and 12 hours)
    valid_replies = reply_df[(reply_df['diff_minutes'] >= 0.08) & (reply_df['diff_minutes'] <= 720)]

    if valid_replies.empty:
        return pd.DataFrame(columns=['User', 'Avg Response Time (mins)', 'Median (mins)', 'Replies'])

    res_df = valid_replies.groupby('user')['diff_minutes'].agg(['mean', 'median', 'count']).reset_index()
    res_df.columns = ['User', 'Avg Response Time (mins)', 'Median (mins)', 'Replies']
    res_df['Avg Response Time (mins)'] = round(res_df['Avg Response Time (mins)'], 1)
    res_df['Median (mins)'] = round(res_df['Median (mins)'], 1)
    res_df.sort_values(by='Avg Response Time (mins)', ascending=True, inplace=True)
    return res_df


# 3. CONVERSATION STARTERS & CLOSERS
def conversation_starters_closers(df, gap_hours=4):
    temp = df[df['user'] != 'group_notification'].copy()
    temp.sort_values('date', inplace=True)
    temp['time_since_prev'] = (temp['date'] - temp['date'].shift(1)).dt.total_seconds() / 3600.0

    # Starters: First message after a silence of >= gap_hours
    starters = temp[temp['time_since_prev'] >= gap_hours]['user'].value_counts().reset_index()
    starters.columns = ['User', 'Conversations Started']

    # Closers: Message sent right before the silence
    closers_indices = temp[temp['time_since_prev'] >= gap_hours].index - 1
    closers_indices = [idx for idx in closers_indices if idx in temp.index]
    closers = temp.loc[closers_indices, 'user'].value_counts().reset_index()
    closers.columns = ['User', 'Conversations Closed (Last Word)']

    return starters, closers


# 4. DOMAIN / URL CATEGORIZER
def extract_top_domains(selected_user, df):
    temp = df.copy()
    if selected_user != 'Overall':
        temp = temp[temp['user'] == selected_user]

    links = []
    for msg in temp['message']:
        links.extend(extract.find_urls(str(msg)))

    domains = []
    for link in links:
        try:
            if not link.startswith('http'):
                link = 'http://' + link
            domain = urlparse(link).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            if domain:
                domains.append(domain)
        except Exception:
            continue

    if not domains:
        return pd.DataFrame(columns=['Domain', 'Count'])

    counts = Counter(domains).most_common(10)
    return pd.DataFrame(counts, columns=['Domain', 'Count'])


# 5. WHATSAPP WRAPPED / FUN AWARDS GENERATOR
def generate_wrapped_awards(df):
    temp = df[df['user'] != 'group_notification'].copy()
    if temp.empty:
        return []

    awards = []

    # 1. 👑 The Yapper (Highest Total Words)
    temp['word_count'] = temp['message'].apply(lambda x: len(str(x).split()))
    user_words = temp.groupby('user')['word_count'].sum()
    if not user_words.empty:
        top_yapper = user_words.idxmax()
        awards.append({
            'icon': '👑',
            'title': 'The Grand Yapper',
            'winner': top_yapper,
            'stat': f"{user_words.max():,} words typed",
            'desc': 'Sent the most total words and never runs out of things to say!'
        })

    # 2. ⚡ Lightning Replier (Fastest Average Response)
    res_df = response_time_analysis(df)
    if not res_df.empty:
        fastest = res_df.iloc[0]
        awards.append({
            'icon': '⚡',
            'title': 'Lightning Responder',
            'winner': fastest['User'],
            'stat': f"{fastest['Avg Response Time (mins)']} mins avg reply",
            'desc': 'Replies faster than humanly possible. Always online!'
        })
        if len(res_df) > 1:
            slowest = res_df.iloc[-1]
            awards.append({
                'icon': '👻',
                'title': 'The Certified Ghost',
                'winner': slowest['User'],
                'stat': f"{slowest['Avg Response Time (mins)']} mins avg reply",
                'desc': 'Takes their sweet time to reply. Probably left on read!'
            })

    # 3. 🌙 Night Owl (Most messages between 12 AM - 5 AM)
    night_msgs = temp[temp['hour'].isin([0, 1, 2, 3, 4])]
    if not night_msgs.empty:
        night_winner = night_msgs['user'].value_counts().idxmax()
        awards.append({
            'icon': '🌙',
            'title': 'Ultimate Night Owl',
            'winner': night_winner,
            'stat': f"{night_msgs['user'].value_counts().max():,} late-night messages",
            'desc': 'Thrives after midnight when the rest of the world sleeps.'
        })

    # 4. 🤣 Laugh Champion
    laugh_pattern = r'(haha|lol|rofl|lmao|hehe|😂|🤣|😹|xd)'
    laugh_msgs = temp[temp['message'].astype(str).str.contains(laugh_pattern, case=False, regex=True, na=False)]
    if not laugh_msgs.empty:
        laugh_winner = laugh_msgs['user'].value_counts().idxmax()
        awards.append({
            'icon': '🤣',
            'title': 'Chief Humor Officer',
            'winner': laugh_winner,
            'stat': f"{laugh_msgs['user'].value_counts().max():,} laughs shared",
            'desc': 'Finds everything hilarious and keeps the group chat energized!'
        })

    # 5. 📸 Media Mogul
    media_pattern = r'<Media omitted>|image omitted|video omitted|sticker omitted'
    media_msgs = temp[temp['message'].astype(str).str.contains(media_pattern, case=False, regex=True, na=False)]
    if not media_msgs.empty:
        media_winner = media_msgs['user'].value_counts().idxmax()
        awards.append({
            'icon': '📸',
            'title': 'Media Mogul',
            'winner': media_winner,
            'stat': f"{media_msgs['user'].value_counts().max():,} media items",
            'desc': 'Communicates exclusively in stickers, memes, and photos.'
        })

    # 6. 🚀 Conversation Igniter
    starters, _ = conversation_starters_closers(df)
    if not starters.empty:
        starter_winner = starters.iloc[0]
        awards.append({
            'icon': '🚀',
            'title': 'Conversation Igniter',
            'winner': starter_winner['User'],
            'stat': f"{starter_winner['Conversations Started']} chats initiated",
            'desc': 'Revives the chat whenever things get quiet.'
        })

    return awards


# ==========================================
# 🛡️ PII DETECTOR & REDACTOR ENGINE
# ==========================================

PII_PATTERNS = {
    'Email Address': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'Phone Number': r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\b(?:\d{3,5}[-.\s]?\d{4,6}|\d{10})\b',
    'UPI / Payment ID': r'\b[a-zA-Z0-9.\-_]{2,50}@(oksbi|okhdfcbank|okaxis|okicici|paytm|ybl|ibl|axl|upi|sbi|hdfcbank|icici|barodampay|fbl|idfcbank|postbank|waaxis|wahdfc|waicici|wasbi)\b',
    'Card / Account Number': r'\b(?:\d{4}[ -]?){3}\d{4}\b',
    'Aadhaar / National ID': r'\b\d{4}\s\d{4}\s\d{4}\b|\b\d{3}-\d{2}-\d{4}\b',
    'PAN Card (India)': r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
    'URL / Link': r'https?://[^\s<>"]+|www\.[^\s<>"]+',
    'IP Address': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
    'PIN / Postal Code': r'\b[1-9][0-9]{2}\s?[0-9]{3}\b'
}

def detect_pii_in_text(text):
    """Scans text and returns a dictionary of detected PII matches by category."""
    detected = []
    text_str = str(text)
    
    # Avoid scanning omitted media tags
    if '<Media omitted>' in text_str or 'This message was deleted' in text_str:
        return detected

    for category, pattern in PII_PATTERNS.items():
        # Special check for phone numbers to avoid matching plain dates or small numbers
        matches = re.finditer(pattern, text_str, flags=re.IGNORECASE)
        for match in matches:
            val = match.group().strip()
            # Filter out non-phone numeric strings (e.g. timestamps like 12:30 or single short numbers)
            if category == 'Phone Number':
                digits_only = re.sub(r'\D', '', val)
                if len(digits_only) < 10 or len(digits_only) > 13:
                    continue
                if ':' in val or '/' in val:
                    continue
            elif category == 'PIN / Postal Code':
                digits_only = re.sub(r'\D', '', val)
                if len(digits_only) != 6:
                    continue

            detected.append({
                'category': category,
                'value': val,
                'start': match.start(),
                'end': match.end()
            })
    return detected


def mask_pii_in_text(text, mask_style="tag"):
    """
    Masks detected PII in text.
    mask_style options:
      - 'tag': [REDACTED: PHONE NUMBER]
      - 'block': ████████
      - 'asterisk': p***@gmail.com / +91 98*****
    """
    text_str = str(text)
    if '<Media omitted>' in text_str or 'This message was deleted' in text_str:
        return text_str

    for category, pattern in PII_PATTERNS.items():
        def replace_fn(match):
            val = match.group()
            if category == 'Phone Number':
                digits_only = re.sub(r'\D', '', val)
                if len(digits_only) < 10 or len(digits_only) > 13 or ':' in val or '/' in val:
                    return val
            elif category == 'PIN / Postal Code':
                digits_only = re.sub(r'\D', '', val)
                if len(digits_only) != 6:
                    return val

            if mask_style == "tag":
                return f"[REDACTED: {category.upper()}]"
            elif mask_style == "block":
                return "█" * max(len(val), 6)
            elif mask_style == "asterisk":
                if len(val) <= 4:
                    return "*" * len(val)
                return val[:2] + ("*" * (len(val) - 4)) + val[-2:]
            return "[REDACTED]"

        text_str = re.sub(pattern, replace_fn, text_str, flags=re.IGNORECASE)
    return text_str


def analyze_pii_in_chat(df):
    """Performs full chat PII detection and aggregates summary metrics."""
    temp = df[df['user'] != 'group_notification'].copy()
    
    all_detections = []
    for idx, row in temp.iterrows():
        matches = detect_pii_in_text(row['message'])
        for m in matches:
            all_detections.append({
                'Date': row['date'],
                'User': row['user'],
                'Category': m['category'],
                'Exposed Value': m['value'],
                'Original Message': row['message']
            })

    if not all_detections:
        return {
            'total_pii_count': 0,
            'category_counts': pd.Series(dtype=int),
            'user_pii_counts': pd.DataFrame(columns=['User', 'PII Shared Count']),
            'pii_records_df': pd.DataFrame(columns=['Date', 'User', 'Category', 'Exposed Value', 'Original Message']),
            'redacted_df': df.copy()
        }

    pii_df = pd.DataFrame(all_detections)
    category_counts = pii_df['Category'].value_counts()
    user_pii_counts = pii_df['User'].value_counts().reset_index()
    user_pii_counts.columns = ['User', 'PII Shared Count']

    # Generate a redacted DataFrame copy
    redacted_df = df.copy()
    redacted_df['message'] = redacted_df['message'].apply(lambda x: mask_pii_in_text(x, mask_style="tag"))

    return {
        'total_pii_count': len(pii_df),
        'category_counts': category_counts,
        'user_pii_counts': user_pii_counts,
        'pii_records_df': pii_df,
        'redacted_df': redacted_df
    }


def extract_contacts_directory(df):
    """
    Scans the chat for all phone numbers and emails, clean-formats them,
    and creates one-click clickable actions:
      - WhatsApp: https://wa.me/...
      - Call: tel:...
      - Email: mailto:...
    Also generates downloadable vCard (.vcf) format.
    """
    temp = df[df['user'] != 'group_notification'].copy()
    contacts = []
    seen = set()

    for _, row in temp.iterrows():
        matches = detect_pii_in_text(row['message'])
        for m in matches:
            cat = m['category']
            val = m['value']

            if cat == 'Phone Number':
                clean_digits = re.sub(r'\D', '', val)
                if len(clean_digits) == 10:
                    # Default to India (+91) if 10-digit number without country code
                    clean_digits = "91" + clean_digits
                key = ('phone', clean_digits)
                if key not in seen:
                    seen.add(key)
                    contacts.append({
                        'Type': '📞 Phone',
                        'Sender': row['user'],
                        'Contact Detail': val,
                        'WhatsApp Link': f"https://wa.me/{clean_digits}",
                        'Call Action': f"tel:+{clean_digits}",
                        'Action Type': 'phone',
                        'Date Shared': row['date'],
                        'Context': row['message']
                    })

            elif cat == 'Email Address':
                clean_email = val.strip().lower()
                key = ('email', clean_email)
                if key not in seen:
                    seen.add(key)
                    contacts.append({
                        'Type': '📧 Email',
                        'Sender': row['user'],
                        'Contact Detail': clean_email,
                        'Email Action': f"mailto:{clean_email}?subject=Connecting%20via%20WhatsApp%20Chat",
                        'Action Type': 'email',
                        'Date Shared': row['date'],
                        'Context': row['message']
                    })

    contacts_df = pd.DataFrame(contacts)
    
    # Generate vCard (.vcf) content for phone contacts
    vcard_entries = []
    for idx, c in enumerate(contacts):
        vcard_entries.append("BEGIN:VCARD")
        vcard_entries.append("VERSION:3.0")
        vcard_entries.append(f"FN:{c['Sender']} (WA Contact {idx+1})")
        if c['Action Type'] == 'phone':
            vcard_entries.append(f"TEL;TYPE=CELL:{c['Contact Detail']}")
        elif c['Action Type'] == 'email':
            vcard_entries.append(f"EMAIL;TYPE=INTERNET:{c['Contact Detail']}")
        vcard_entries.append(f"NOTE:Extracted from WhatsApp Chat on {c['Date Shared']}")
        vcard_entries.append("END:VCARD\n")
    
    vcard_data = "\n".join(vcard_entries)
    return contacts_df, vcard_data


def get_group_members_roster(df):
    """
    Extracts all participants involved in the chat group along with their
    message count, first/last active timestamps, and identifies if the user is a phone number.
    """
    temp = df[df['user'] != 'group_notification'].copy()
    if temp.empty:
        return pd.DataFrame()

    roster = []
    for user_name, group in temp.groupby('user'):
        total_msgs = len(group)
        total_words = sum(len(str(m).split()) for m in group['message'])
        first_seen = group['date'].min()
        last_seen = group['date'].max()

        # Check if the participant name is an unsaved phone number (e.g., +91 98765..., 98765...)
        clean_digits = re.sub(r'\D', '', str(user_name))
        is_phone_handle = len(clean_digits) >= 10 and len(clean_digits) <= 13
        if is_phone_handle and len(clean_digits) == 10:
            clean_digits = "91" + clean_digits

        wa_link = f"https://wa.me/{clean_digits}" if is_phone_handle else None

        roster.append({
            'Member': user_name,
            'Is Phone Number': '📱 Unsaved Phone' if is_phone_handle else '👤 Saved Name',
            'Messages Sent': total_msgs,
            'Words Typed': total_words,
            'First Active': first_seen.strftime('%d %b %Y, %I:%M %p'),
            'Last Active': last_seen.strftime('%d %b %Y, %I:%M %p'),
            'WhatsApp Direct Link': wa_link
        })

    roster_df = pd.DataFrame(roster)
    roster_df.sort_values(by='Messages Sent', ascending=False, inplace=True)
    return roster_df




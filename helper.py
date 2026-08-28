import os
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import emoji

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
import re
import pandas as pd


def preprocess(data):
    # Normalize unicode non-breaking spaces and narrow spaces
    data = data.replace('\u202f', ' ').replace('\xa0', ' ').replace('\u200e', '')

    # Common WhatsApp datetime regex patterns
    patterns = [
        # Android 12-hr format: "12/31/22, 11:59 PM - " or "31/12/2022, 11:59 pm - "
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s[aApP][mM])\s-\s',
        # Android 24-hr format: "31/12/2022, 23:59 - "
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?)\s-\s',
        # iOS format with brackets: "[31/12/22, 11:59:59 PM] "
        r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?(?:\s[aApP][mM])?)\]\s',
    ]

    matched_pattern = None
    for pat in patterns:
        if re.search(pat, data):
            matched_pattern = pat
            break

    if not matched_pattern:
        matched_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[aApP][mM])\s-\s'

    messages = re.split(matched_pattern, data)

    if len(messages) <= 1:
        return pd.DataFrame()

    dates = messages[1::2]
    user_messages = messages[2::2]

    df = pd.DataFrame({'user_message': user_messages, 'message_date': dates})

    df['date'] = pd.to_datetime(df['message_date'], errors='coerce', format='mixed')
    df.dropna(subset=['date'], inplace=True)

    if df.empty:
        return pd.DataFrame()

    users = []
    cleaned_messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) >= 3:
            users.append(entry[1].strip())
            cleaned_messages.append(entry[2])
        else:
            users.append('group_notification')
            cleaned_messages.append(entry[0])

    df['user'] = users
    df['message'] = cleaned_messages
    df.drop(columns=['user_message', 'message_date'], inplace=True, errors='ignore')

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append("23-00")
        elif hour == 0:
            period.append("00-01")
        else:
            period.append(f"{hour:02d}-{(hour + 1):02d}")
    df['period'] = period

    return df
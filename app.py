import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import preprocessor
import helper

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="WhatsApp Chat Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN STYLING ---
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(37, 211, 102, 0.4);
    }
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #25D366;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Section Headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 20px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    /* App Header */
    .header-box {
        background: linear-gradient(90deg, #128C7E 0%, #075E54 100%);
        padding: 20px 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
        <span style="font-size: 2.2rem;">💬</span>
        <div>
            <h2 style="margin: 0; padding: 0; font-size: 1.4rem;">WhatsApp Analyzer</h2>
            <p style="margin: 0; color: #888; font-size: 0.8rem;">Smart Chat Insights & Analytics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload exported chat (.txt)",
        type=["txt"],
        help="Export your chat from WhatsApp (without media) and upload the .txt file here."
    )

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)
    except Exception as e:
        st.error(f"Error parsing chat file: {str(e)}")
        df = pd.DataFrame()

    if df.empty:
        st.error("⚠️ Could not parse messages from this file. Please ensure it's a standard WhatsApp export file.")
    else:
        # Sidebar User Filter
        user_list = [u for u in df['user'].unique().tolist() if u != 'group_notification']
        user_list.sort()
        user_list.insert(0, "Overall")

        with st.sidebar:
            st.divider()
            selected_user = st.selectbox("👤 Select Participant", user_list)
            
            # Date Range Filter
            min_date = df['only_date'].min()
            max_date = df['only_date'].max()
            if min_date != max_date:
                date_range = st.date_input(
                    "📅 Date Range Filter",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) == 2:
                    df = df[(df['only_date'] >= date_range[0]) & (df['only_date'] <= date_range[1])]

            st.divider()
            st.subheader("🛡️ Privacy & PII Guard")
            mask_pii_enabled = st.checkbox(
                "Anonymize / Mask PII Data",
                value=False,
                help="Automatically masks phone numbers, emails, URLs, payment IDs, and account numbers across the dashboard."
            )
            mask_style = "tag"
            if mask_pii_enabled:
                mask_choice = st.radio(
                    "Masking Style",
                    ["[TAG] [REDACTED: TYPE]", "Asterisk (p***@gmail.com)", "Solid Block (██████)"],
                    index=0
                )
                if "[TAG]" in mask_choice:
                    mask_style = "tag"
                elif "Asterisk" in mask_choice:
                    mask_style = "asterisk"
                else:
                    mask_style = "block"

        # Pre-compute PII Analysis on unmasked data before optional visual masking
        raw_df_for_pii = df.copy()
        pii_summary = helper.analyze_pii_in_chat(raw_df_for_pii)

        # If user toggles global anonymizer, mask messages in working df
        if mask_pii_enabled:
            df['message'] = df['message'].apply(lambda x: helper.mask_pii_in_text(x, mask_style=mask_style))

        # --- HEADER BANNER ---
        st.markdown(f"""
        <div class="header-box">
            <h2 style="margin: 0; padding: 0;">📊 Chat Analysis Dashboard</h2>
            <p style="margin: 4px 0 0 0; opacity: 0.9;">
                Analyzing <b>{selected_user}</b> &bull; {df['only_date'].min().strftime('%d %b %Y')} to {df['only_date'].max().strftime('%d %b %Y')}
                {' &bull; <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 6px; font-size: 0.8rem;">🛡️ PII Masked Mode</span>' if mask_pii_enabled else ''}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- TOP KPI METRICS ---
        raw_stats = helper.fetch_stats(selected_user, df)
        if isinstance(raw_stats, dict):
            stats = raw_stats
        else:
            num_messages, words, num_media_messages, num_links = raw_stats
            stats = {
                'num_messages': num_messages,
                'num_words': words,
                'num_media': num_media_messages,
                'num_links': num_links,
                'num_emojis': 0,
                'avg_words_per_msg': round(words / num_messages, 1) if num_messages > 0 else 0,
                'active_days': df['only_date'].nunique() if 'only_date' in df else 0
            }

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💬</div>
                <div class="metric-value">{stats['num_messages']:,}</div>
                <div class="metric-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📝</div>
                <div class="metric-value">{stats['num_words']:,}</div>
                <div class="metric-label">Total Words</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🖼️</div>
                <div class="metric-value">{stats['num_media']:,}</div>
                <div class="metric-label">Media Shared</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔗</div>
                <div class="metric-value">{stats['num_links']:,}</div>
                <div class="metric-label">Links Shared</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">😀</div>
                <div class="metric-value">{stats['num_emojis']:,}</div>
                <div class="metric-label">Emojis Used</div>
            </div>
            """, unsafe_allow_html=True)
        with col6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⚡</div>
                <div class="metric-value">{stats['avg_words_per_msg']}</div>
                <div class="metric-label">Avg Words / Msg</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- INTERACTIVE TABS ---
        tab_wrapped, tab_trends, tab_users, tab_activity, tab_sentiment, tab_words_emoji, tab_pii, tab_explorer = st.tabs([
            "🎁 WhatsApp Wrapped",
            "📈 Timelines & Trends",
            "👥 Dynamics & Speed",
            "🕒 Activity Patterns",
            "🎭 Sentiment & Mood",
            "🔤 Words, Emojis & Links",
            "🛡️ PII & Privacy Guard",
            "🔍 Chat Explorer"
        ])

        # === TAB 1: WHATSAPP WRAPPED / AWARDS ===
        with tab_wrapped:
            st.subheader("🎉 Group Chat Wrapped & Superlatives")
            st.caption("Fun personality badges and superlatives computed from chat behavior")

            awards = helper.generate_wrapped_awards(df)
            if awards:
                # Render award cards in grid of 3
                for i in range(0, len(awards), 3):
                    cols = st.columns(3)
                    for j, award in enumerate(awards[i:i+3]):
                        with cols[j]:
                            st.markdown(f"""
                            <div class="metric-card" style="text-align: left; padding: 22px; margin-bottom: 16px; border-left: 4px solid #25D366;">
                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                                    <span style="font-size: 2.2rem;">{award['icon']}</span>
                                    <span style="background: rgba(37, 211, 102, 0.15); color: #25D366; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">AWARD</span>
                                </div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 2px;">{award['title']}</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #25D366; margin-bottom: 6px;">{award['winner']}</div>
                                <div style="font-size: 0.85rem; color: #E2E8F0; font-weight: 500; margin-bottom: 6px;">📊 {award['stat']}</div>
                                <div style="font-size: 0.8rem; color: #A0AEC0; line-height: 1.3;">{award['desc']}</div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("Not enough data to calculate Wrapped awards.")

        # === TAB 2: TIMELINES & TRENDS ===
        with tab_trends:
            col_t1, col_t2 = st.columns([1, 1])

            # Monthly Timeline
            with col_t1:
                st.subheader("📅 Monthly Message Volume")
                monthly_df = helper.monthly_timeline(selected_user, df)
                if not monthly_df.empty:
                    fig_monthly = px.area(
                        monthly_df,
                        x="time",
                        y="message",
                        labels={"time": "Month", "message": "Messages"},
                        color_discrete_sequence=["#25D366"]
                    )
                    fig_monthly.update_traces(mode="lines+markers", line=dict(width=2))
                    fig_monthly.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_monthly, use_container_width=True)
                else:
                    st.info("No monthly timeline data available.")

            # Daily Timeline
            with col_t2:
                st.subheader("📈 Daily Message Activity")
                daily_df = helper.daily_timeline(selected_user, df)
                if not daily_df.empty:
                    fig_daily = px.line(
                        daily_df,
                        x="date",
                        y="messages",
                        labels={"date": "Date", "messages": "Messages"},
                        color_discrete_sequence=["#00C49F"]
                    )
                    fig_daily.update_layout(
                        template="plotly_dark",
                        height=380,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis=dict(rangeslider=dict(visible=True), type="date")
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
                else:
                    st.info("No daily timeline data available.")

        # === TAB 3: DYNAMICS & SPEED ===
        with tab_users:
            if selected_user == "Overall":
                col_u1, col_u2 = st.columns([1, 1])
                x, percent_df = helper.most_busy_users(df)

                with col_u1:
                    st.subheader("🏆 Top Contributors (Messages)")
                    fig_users = px.bar(
                        x=x.index,
                        y=x.values,
                        labels={"x": "User", "y": "Messages"},
                        color=x.values,
                        color_continuous_scale="Viridis"
                    )
                    fig_users.update_layout(template="plotly_dark", height=380, coloraxis_showscale=False)
                    st.plotly_chart(fig_users, use_container_width=True)

                with col_u2:
                    st.subheader("🥧 Contribution Share")
                    fig_pie = px.pie(
                        percent_df.head(8),
                        values="Percentage (%)",
                        names="User",
                        hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(template="plotly_dark", height=380, showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.divider()

                # Response Time & Ghosting Analysis
                st.subheader("⚡ Response Speed & Reply Delays")
                st.caption("Calculates how quickly participants respond when replying to someone else")
                res_df = helper.response_time_analysis(df)

                if not res_df.empty:
                    col_r1, col_r2 = st.columns([1, 1])
                    with col_r1:
                        fig_res = px.bar(
                            res_df.head(10),
                            x="Avg Response Time (mins)",
                            y="User",
                            orientation="h",
                            labels={"Avg Response Time (mins)": "Average Reply Delay (minutes)", "User": "User"},
                            color="Avg Response Time (mins)",
                            color_continuous_scale="Turbo"
                        )
                        fig_res.update_layout(template="plotly_dark", height=350, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                        st.plotly_chart(fig_res, use_container_width=True)

                    with col_r2:
                        st.dataframe(res_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough consecutive replies between participants to calculate response times.")

                st.divider()

                # Conversation Starters vs Closers
                st.subheader("🚀 Conversation Starters vs Closers")
                st.caption("Identifies who starts chatting after long silences (> 4 hours) vs who sends the closing message")
                starters, closers = helper.conversation_starters_closers(df, gap_hours=4)

                col_s1, col_s2 = st.columns([1, 1])
                with col_s1:
                    if not starters.empty:
                        fig_s = px.bar(
                            starters.head(8),
                            x="User",
                            y="Conversations Started",
                            title="Conversations Initiated",
                            color="Conversations Started",
                            color_continuous_scale="Mint"
                        )
                        fig_s.update_layout(template="plotly_dark", height=320, coloraxis_showscale=False)
                        st.plotly_chart(fig_s, use_container_width=True)
                with col_s2:
                    if not closers.empty:
                        fig_c = px.bar(
                            closers.head(8),
                            x="User",
                            y="Conversations Closed (Last Word)",
                            title="Conversations Closed (Last Word)",
                            color="Conversations Closed (Last Word)",
                            color_continuous_scale="Burg"
                        )
                        fig_c.update_layout(template="plotly_dark", height=320, coloraxis_showscale=False)
                        st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info(f"Currently viewing individual stats for **{selected_user}**. Switch to **Overall** in the sidebar to compare member dynamics and response times.")

        # === TAB 4: ACTIVITY PATTERNS ===
        with tab_activity:
            col_a1, col_a2 = st.columns([1, 1])

            with col_a1:
                st.subheader("📅 Activity by Day of Week")
                busy_day = helper.week_activity_map(selected_user, df)
                fig_day = px.bar(
                    x=busy_day.index,
                    y=busy_day.values,
                    labels={"x": "Day", "y": "Messages"},
                    color=busy_day.values,
                    color_continuous_scale="Purples"
                )
                fig_day.update_layout(template="plotly_dark", height=380, coloraxis_showscale=False)
                st.plotly_chart(fig_day, use_container_width=True)

            with col_a2:
                st.subheader("🗓️ Activity by Month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig_month = px.bar(
                    x=busy_month.index,
                    y=busy_month.values,
                    labels={"x": "Month", "y": "Messages"},
                    color=busy_month.values,
                    color_continuous_scale="Sunset"
                )
                fig_month.update_layout(template="plotly_dark", height=380, coloraxis_showscale=False)
                st.plotly_chart(fig_month, use_container_width=True)

            st.subheader("🔥 Hourly Activity Heatmap")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            if not user_heatmap.empty:
                fig_hm = px.imshow(
                    user_heatmap,
                    labels=dict(x="Time of Day (Hours)", y="Day of Week", color="Messages"),
                    x=user_heatmap.columns,
                    y=user_heatmap.index,
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                fig_hm.update_layout(template="plotly_dark", height=360)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.warning("Not enough data to construct an activity heatmap.")

        # === TAB 5: SENTIMENT & MOOD ===
        with tab_sentiment:
            st.subheader("🎭 Chat Sentiment & Tone Analysis")
            st.caption("Powered by VADER (Valence Aware Dictionary and sEntiment Reasoner)")

            sentiment_data = helper.analyze_sentiment(selected_user, df)
            if sentiment_data is not None:
                col_sen1, col_sen2 = st.columns([1, 1])

                with col_sen1:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 20px;">
                        <div style="font-size: 0.9rem; color: #888; text-transform: uppercase;">Average Polarity Score</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {'#25D366' if sentiment_data['avg_compound'] >= 0.05 else ('#FF5252' if sentiment_data['avg_compound'] <= -0.05 else '#FFD700')};">
                            {sentiment_data['avg_compound']:+}
                        </div>
                        <div style="font-size: 0.85rem; color: #aaa;">Scale: -1.0 (Very Negative) to +1.0 (Very Positive)</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Sentiment Pie Chart
                    counts = sentiment_data['counts']
                    fig_sent_pie = px.pie(
                        values=counts.values,
                        names=counts.index,
                        hole=0.45,
                        color=counts.index,
                        color_discrete_map={
                            'Positive': '#25D366',
                            'Neutral': '#64748B',
                            'Negative': '#EF4444'
                        },
                        title="Sentiment Distribution"
                    )
                    fig_sent_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_sent_pie.update_layout(template="plotly_dark", height=320)
                    st.plotly_chart(fig_sent_pie, use_container_width=True)

                with col_sen2:
                    if selected_user == "Overall" and sentiment_data['user_sentiment'] is not None:
                        st.subheader("😊 Participant Sentiment Ranking")
                        st.dataframe(sentiment_data['user_sentiment'], use_container_width=True, hide_index=True)
                    else:
                        st.caption("Sample Message Sentiment Breakdown")
                        sample_sent = sentiment_data['sentiment_df'][['date', 'user', 'message', 'sentiment', 'compound']].head(20)
                        sample_sent.rename(columns={'compound': 'Score'}, inplace=True)
                        st.dataframe(sample_sent, use_container_width=True, hide_index=True)
            else:
                st.info("No text messages available for sentiment evaluation.")

        # === TAB 6: WORDS, EMOJIS & LINKS ===
        with tab_words_emoji:
            col_w1, col_w2 = st.columns([1, 1])

            with col_w1:
                st.subheader("☁️ Chat Word Cloud")
                df_wc = helper.create_wordcloud(selected_user, df)
                if df_wc is not None:
                    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0e1117')
                    ax.imshow(df_wc, interpolation='bilinear')
                    ax.axis("off")
                    plt.tight_layout(pad=0)
                    st.pyplot(fig)
                else:
                    st.info("No enough words found for WordCloud.")

            with col_w2:
                st.subheader("🔠 Most Common Words")
                most_common_df = helper.most_common_words(selected_user, df)
                if not most_common_df.empty:
                    fig_words = px.bar(
                        most_common_df.head(15),
                        x="count",
                        y="word",
                        orientation="h",
                        labels={"count": "Frequency", "word": "Word"},
                        color="count",
                        color_continuous_scale="Tealgrn"
                    )
                    fig_words.update_layout(
                        template="plotly_dark",
                        height=400,
                        yaxis=dict(autorange="reversed"),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_words, use_container_width=True)
                else:
                    st.info("No common words found.")

            st.divider()

            # Emoji Section
            st.subheader("😀 Emoji Breakdown")
            emoji_df = helper.emoji_helper(selected_user, df)

            if not emoji_df.empty:
                col_e1, col_e2 = st.columns([1, 1])
                with col_e1:
                    fig_emoji = px.pie(
                        emoji_df.head(8),
                        values="count",
                        names="emoji",
                        hole=0.4,
                        title="Top Emojis Share"
                    )
                    fig_emoji.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont_size=16
                    )
                    fig_emoji.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_emoji, use_container_width=True)

                with col_e2:
                    fig_bar_emoji = px.bar(
                        emoji_df.head(10),
                        x="emoji",
                        y="count",
                        labels={"emoji": "Emoji", "count": "Occurrences"},
                        color="count",
                        color_continuous_scale="Plasma"
                    )
                    fig_bar_emoji.update_layout(
                        template="plotly_dark",
                        height=350,
                        coloraxis_showscale=False,
                        xaxis=dict(tickfont=dict(size=18))
                    )
                    st.plotly_chart(fig_bar_emoji, use_container_width=True)
            else:
                st.info("No emojis detected in this chat.")

            st.divider()

            # Web Domains / Links Section
            st.subheader("🔗 Top Shared Domains & Websites")
            domain_df = helper.extract_top_domains(selected_user, df)
            if not domain_df.empty:
                fig_domains = px.bar(
                    domain_df,
                    x="Count",
                    y="Domain",
                    orientation="h",
                    color="Count",
                    color_continuous_scale="Burgyl"
                )
                fig_domains.update_layout(template="plotly_dark", height=320, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                st.plotly_chart(fig_domains, use_container_width=True)
            else:
                st.info("No external web links detected in this chat.")

        # === TAB 7: PII & PRIVACY GUARD ===
        with tab_pii:
            st.subheader("🛡️ Sensitive Data (PII) Audit & Protection")
            st.caption("Automatically scans conversations for exposed phone numbers, email addresses, payment IDs, URLs, and account numbers.")

            total_pii = pii_summary['total_pii_count']
            col_p1, col_p2, col_p3 = st.columns(3)

            with col_p1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {'#EF4444' if total_pii > 0 else '#25D366'};">
                    <div class="metric-icon">🔍</div>
                    <div class="metric-value" style="color: {'#EF4444' if total_pii > 0 else '#25D366'};">{total_pii:,}</div>
                    <div class="metric-label">Total PII Entities Found</div>
                </div>
                """, unsafe_allow_html=True)
            with col_p2:
                categories_found = len(pii_summary['category_counts'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🗂️</div>
                    <div class="metric-value">{categories_found}</div>
                    <div class="metric-label">PII Categories Detected</div>
                </div>
                """, unsafe_allow_html=True)
            with col_p3:
                masked_status = "✅ ACTIVE" if mask_pii_enabled else "⚠️ OFF"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🛡️</div>
                    <div class="metric-value" style="color: {'#25D366' if mask_pii_enabled else '#F59E0B'};">{masked_status}</div>
                    <div class="metric-label">Dashboard Masking Status</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if total_pii > 0:
                col_chart1, col_chart2 = st.columns([1, 1])

                with col_chart1:
                    st.subheader("📊 PII Breakdown by Category")
                    cat_counts = pii_summary['category_counts']
                    fig_pii_pie = px.pie(
                        values=cat_counts.values,
                        names=cat_counts.index,
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig_pii_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pii_pie.update_layout(template="plotly_dark", height=340)
                    st.plotly_chart(fig_pii_pie, use_container_width=True)

                with col_chart2:
                    st.subheader("👤 Senders with Most Exposed PII")
                    user_pii = pii_summary['user_pii_counts'].head(8)
                    fig_pii_user = px.bar(
                        user_pii,
                        x="User",
                        y="PII Shared Count",
                        color="PII Shared Count",
                        color_continuous_scale="Reds"
                    )
                    fig_pii_user.update_layout(template="plotly_dark", height=340, coloraxis_showscale=False)
                    st.plotly_chart(fig_pii_user, use_container_width=True)

                st.divider()

                # Detailed PII Audit Records
                st.subheader("📋 Detected Sensitive Information Log")
                st.caption("Review exposed values before masking or sharing with third parties.")
                records_df = pii_summary['pii_records_df'].copy()
                
                # Option to toggle visibility of raw values inside this tab
                show_unmasked = st.checkbox("👁️ Reveal unmasked sensitive values in table below", value=False)
                if not show_unmasked:
                    records_df['Exposed Value'] = records_df['Exposed Value'].apply(lambda x: helper.mask_pii_in_text(x, mask_style="asterisk"))
                    records_df['Original Message'] = records_df['Original Message'].apply(lambda x: helper.mask_pii_in_text(x, mask_style="tag"))

                st.dataframe(records_df, use_container_width=True, hide_index=True)

                st.divider()

                # Export Fully Redacted Clean Chat
                st.subheader("📥 Export Privacy-Sanitized Chat")
                st.write("Generate a clean version of this chat file with all phone numbers, emails, IDs, and financial tokens completely redacted.")

                clean_mask_style = st.selectbox(
                    "Select Export Masking Format",
                    ["[TAG] e.g. [REDACTED: PHONE NUMBER]", "Solid Block e.g. ████████", "Partial Asterisk e.g. +91 98****3210"],
                    index=0
                )
                export_style = "tag" if "[TAG]" in clean_mask_style else ("block" if "Solid" in clean_mask_style else "asterisk")

                sanitized_df = df.copy()
                sanitized_df['message'] = sanitized_df['message'].apply(lambda x: helper.mask_pii_in_text(x, mask_style=export_style))
                
                # Format as downloadable .txt file
                txt_lines = []
                for _, row in sanitized_df.iterrows():
                    txt_lines.append(f"{row['date'].strftime('%m/%d/%y, %I:%M %p')} - {row['user']}: {row['message']}")
                sanitized_txt = "\n".join(txt_lines)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="🛡️ Download Redacted Chat (.txt)",
                        data=sanitized_txt.encode('utf-8'),
                        file_name="sanitized_whatsapp_chat.txt",
                        mime="text/plain"
                    )
                with col_dl2:
                    csv_sanitized = sanitized_df[['date', 'user', 'message']].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📊 Download Redacted Chat (.csv)",
                        data=csv_sanitized,
                        file_name="sanitized_whatsapp_chat.csv",
                        mime="text/csv"
                    )
            else:
                st.success("🎉 No sensitive PII entities (phone numbers, emails, IDs, payment tokens) were detected in this chat.")

        # === TAB 8: CHAT EXPLORER ===
        with tab_explorer:
            st.subheader("🔍 Message Search & Chat Logs")
            search_query = st.text_input("Search messages by keyword:", placeholder="e.g. happy, project, meeting...")

            filtered_df = df[df['user'] != 'group_notification']
            if selected_user != 'Overall':
                filtered_df = filtered_df[filtered_df['user'] == selected_user]

            if search_query:
                filtered_df = filtered_df[filtered_df['message'].astype(str).str.contains(search_query, case=False, na=False)]

            st.caption(f"Showing **{len(filtered_df):,}** messages")
            display_df = filtered_df[['date', 'user', 'message']].rename(columns={
                'date': 'Timestamp',
                'user': 'Sender',
                'message': 'Message'
            })

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Export Cleaned Data to CSV
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Messages as CSV",
                data=csv_data,
                file_name="whatsapp_chat_analysis.csv",
                mime="text/csv"
            )

else:
    # Empty State Welcome Screen
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h1 style="color: #25D366; font-size: 3rem;">💬 WhatsApp Chat Intelligence</h1>
        <p style="font-size: 1.2rem; color: #888; max-width: 600px; margin: 0 auto 30px auto;">
            Upload your exported WhatsApp chat history from the sidebar to visualize conversations, patterns, and insights.
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div class="metric-card" style="width: 220px;">
                <div class="metric-icon">📈</div>
                <div style="font-weight: 600; margin-bottom: 4px;">Trends & Timelines</div>
                <div style="font-size: 0.8rem; color: #888;">Daily & monthly message evolution</div>
            </div>
            <div class="metric-card" style="width: 220px;">
                <div class="metric-icon">👥</div>
                <div style="font-weight: 600; margin-bottom: 4px;">User Insights</div>
                <div style="font-size: 0.8rem; color: #888;">Contributions & engagement shares</div>
            </div>
            <div class="metric-card" style="width: 220px;">
                <div class="metric-icon">🔥</div>
                <div style="font-weight: 600; margin-bottom: 4px;">Activity Heatmaps</div>
                <div style="font-size: 0.8rem; color: #888;">Peak chatting hours & active days</div>
            </div>
            <div class="metric-card" style="width: 220px;">
                <div class="metric-icon">😀</div>
                <div style="font-weight: 600; margin-bottom: 4px;">Emoji & Word Cloud</div>
                <div style="font-size: 0.8rem; color: #888;">Interactive word & emoji trends</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
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

        # --- HEADER BANNER ---
        st.markdown(f"""
        <div class="header-box">
            <h2 style="margin: 0; padding: 0;">📊 Chat Analysis Dashboard</h2>
            <p style="margin: 4px 0 0 0; opacity: 0.9;">
                Analyzing <b>{selected_user}</b> &bull; {df['only_date'].min().strftime('%d %b %Y')} to {df['only_date'].max().strftime('%d %b %Y')}
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
        tab_trends, tab_users, tab_activity, tab_words_emoji, tab_explorer = st.tabs([
            "📈 Timelines & Trends",
            "👥 Participants",
            "🕒 Activity Patterns",
            "🔤 Words & Emojis",
            "🔍 Chat Explorer"
        ])

        # === TAB 1: TIMELINES & TRENDS ===
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

        # === TAB 2: PARTICIPANTS ===
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
                    fig_users.update_layout(template="plotly_dark", height=400, coloraxis_showscale=False)
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
                    fig_pie.update_layout(template="plotly_dark", height=400, showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("📋 Participant Summary Table")
                st.dataframe(percent_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"You have currently selected **{selected_user}**. Switch to **Overall** in the sidebar to see comparative participant metrics.")

        # === TAB 3: ACTIVITY PATTERNS ===
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

        # === TAB 4: WORDS & EMOJIS ===
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

            # Emoji Section with Native Unicode Browser Rendering (fixes missing box fonts!)
            st.subheader("😀 Emoji Analysis")
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
                    fig_emoji.update_layout(template="plotly_dark", height=380)
                    st.plotly_chart(fig_emoji, use_container_width=True)

                with col_e2:
                    st.caption("Top 10 Emojis Frequency")
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
                        height=380,
                        coloraxis_showscale=False,
                        xaxis=dict(tickfont=dict(size=18))
                    )
                    st.plotly_chart(fig_bar_emoji, use_container_width=True)
            else:
                st.info("No emojis detected in this chat.")

        # === TAB 5: CHAT EXPLORER ===
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
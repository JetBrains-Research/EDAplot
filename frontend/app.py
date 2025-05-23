import streamlit as st

import frontend.vega_chat

prototype_page = st.Page(frontend.vega_chat.main, title="Vega-Lite Chat", url_path="chat", default=True)
# eval_report_page = st.Page("vega_chat_benchmark_report.py", title="Evaluation Report")

# Hack to keep widget state when switching pages:
# https://docs.streamlit.io/develop/concepts/multipage-apps/widgets
for mod in (frontend.vega_chat,):
    for var in dir(mod):
        if var.startswith("WKEY_"):
            key = getattr(mod, var)
            if key in st.session_state:
                st.session_state[key] = st.session_state[key]

pg = st.navigation([prototype_page])
pg.run()

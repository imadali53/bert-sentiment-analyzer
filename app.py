
import streamlit as st
import json
import os
from transformers import pipeline
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def build_agent():
    sentiment_pipe = load_sentiment_model()

    @tool
    def text_summarizer(text: str) -> str:
        """Summarizes the given customer review into one concise sentence."""
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response = llm.invoke(f"Summarize this review in one sentence: {text}")
        return response.content

    @tool
    def keyword_extractor(text: str) -> str:
        """Extracts 3-5 key topics or themes from the customer review."""
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response = llm.invoke(
            f"Extract 3-5 keywords from this review. Return ONLY a comma-separated list: {text}"
        )
        return response.content

    @tool
    def sentiment_analyzer(text: str) -> str:
        """Runs DistilBERT sentiment analysis and returns label + confidence."""
        result = sentiment_pipe(text[:512])[0]
        return f"{result['label']} ({result['score']*100:.2f}%)"

    tools = [text_summarizer, keyword_extractor, sentiment_analyzer]
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]

    SYSTEM_PROMPT = """You are a restaurant review analyst.
When given a customer review, you MUST call all three tools:
1. text_summarizer — to summarize the review
2. keyword_extractor — to extract keywords
3. sentiment_analyzer — to get BERT sentiment

After receiving all three tool results, respond ONLY with valid JSON in this exact schema:
{
  "summary": "one sentence summary here",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "sentiment": "POSITIVE or NEGATIVE",
  "confidence": 98.89,
  "health_label": "Customer Favorite / Needs Attention / Urgent Attention Required"
}

health_label rules:
- POSITIVE + confidence > 80%  → "Customer Favorite"
- POSITIVE + confidence 50-80% → "Needs Attention"
- NEGATIVE + confidence 50-80% → "Needs Attention"
- NEGATIVE + confidence > 80%  → "Urgent Attention Required"

Return ONLY the JSON. No extra text. No markdown fences."""

    def agent_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    tool_map = {t.name: t for t in tools}

    def tool_node(state: AgentState):
        last_message = state["messages"][-1]
        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_fn = tool_map[tool_call["name"]]
            result = tool_fn.invoke(tool_call["args"])
            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
        return {"messages": tool_results}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()

def get_health_color(health_label: str) -> str:
    mapping = {
        "Customer Favorite":         "#2ecc71",
        "Needs Attention":           "#f39c12",
        "Urgent Attention Required": "#e74c3c",
    }
    return mapping.get(health_label, "#95a5a6")

def run_agent(review_text: str) -> dict:
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=review_text)]})
    final_message = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_message = msg
            break
    if final_message is None:
        return {"error": "Agent did not return a final response."}
    try:
        raw = final_message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"error": f"JSON parse failed. Raw output: {final_message.content}"}

st.set_page_config(page_title="Review Intelligence Agent", page_icon="🤖", layout="wide")
st.title("🤖 Restaurant Review Intelligence Agent")
st.caption("Powered by LangGraph + DistilBERT + Groq | Built by Imad Ali")

review = st.text_area(
    "Paste your customer review here:",
    height=150,
    placeholder="e.g. The sajji at this Peshawar dhaba was completely dry and overcooked..."
)

if st.button("🔍 Analyze with AI Agent", use_container_width=True):
    if not review.strip():
        st.warning("Please paste a review before analyzing.")
    else:
        with st.spinner("Agent is running all three tools... please wait ⏳"):
            output = run_agent(review)
        if "error" in output:
            st.error(output["error"])
        else:
            st.success("Analysis complete!")
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### 📝 Summary")
                st.info(output.get("summary", "N/A"))
            with col2:
                st.markdown("### 🔑 Keywords")
                keywords = output.get("keywords", [])
                for kw in keywords:
                    st.markdown(f"- {kw}")
            with col3:
                st.markdown("### 🎯 Sentiment Health")
                label = output.get("health_label", "Unknown")
                confidence = output.get("confidence", 0)
                sentiment = output.get("sentiment", "")
                color = get_health_color(label)
                st.markdown(
                    f"""
                    <div style='background-color:{color}; padding:20px;
                                border-radius:10px; text-align:center;'>
                        <h3 style='color:white; margin:0;'>{label}</h3>
                        <p style='color:white; margin:4px 0;'>{sentiment} — {confidence:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

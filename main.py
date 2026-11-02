import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ROWS = 200
MAX_ATTEMPTS = 3  # SQL generation retries when a query fails
HISTORY_TURNS = 3  # previous exchanges sent to the model for follow-ups
SAMPLE_ROWS = 3
TABLE_SELECT_THRESHOLD = 15  # above this many tables, pick relevant ones first
QUERY_TIMEOUT_MS = 10_000

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "query_log.jsonl"
DESCRIPTIONS_FILE = BASE_DIR / "table_descriptions.json"  # optional {"table": "what it holds"}

EXAMPLE_QUESTIONS = [
    "Which five actors appear in the most films?",
    "How many films are in each category?",
    "Top 10 customers by total payments",
    "Which store has the highest revenue?",
    "Which films have never been rented?",
]

st.set_page_config(page_title="AskDB", layout="wide")
st.title("AskDB")
st.caption("Ask questions about your MySQL database in plain English.")


# ---------- database ----------
def get_engine(host, port, user, password, database):
    url = URL.create(
        "mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )
    return create_engine(url, pool_pre_ping=True, connect_args={"read_timeout": 30})


def load_descriptions() -> dict:
    try:
        return json.loads(DESCRIPTIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_metadata(engine) -> dict:
    """Cheap per-table metadata: columns, primary key, foreign keys, description."""
    insp = inspect(engine)
    descriptions = load_descriptions()
    meta = {}
    for table in insp.get_table_names():
        meta[table] = {
            "columns": [(c["name"], str(c["type"])) for c in insp.get_columns(table)],
            "pk": insp.get_pk_constraint(table).get("constrained_columns", []),
            "fks": [
                (fk["constrained_columns"], fk["referred_table"], fk["referred_columns"])
                for fk in insp.get_foreign_keys(table)
            ],
            "description": descriptions.get(table, ""),
        }
    return meta


def sample_rows(engine, table: str) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT * FROM `{table}` LIMIT {SAMPLE_ROWS}")).fetchall()
    return [" | ".join(str(v)[:40] for v in row) for row in rows]


def describe_table(engine, table: str, info: dict) -> str:
    cols = ", ".join(f"{n} {t}" for n, t in info["columns"])
    lines = [f"TABLE {table}({cols})"]
    if info["description"]:
        lines.append(f"  description: {info['description']}")
    if info["pk"]:
        lines.append(f"  primary key: {', '.join(info['pk'])}")
    for cols_, ref_table, ref_cols in info["fks"]:
        lines.append(f"  foreign key: {', '.join(cols_)} -> {ref_table}({', '.join(ref_cols)})")
    samples = st.session_state.sample_cache.get(table)
    if samples is None:
        try:
            samples = sample_rows(engine, table)
        except Exception:
            samples = []
        st.session_state.sample_cache[table] = samples
    if samples:
        lines.append("  sample rows: " + " || ".join(samples))
    return "\n".join(lines)


def select_tables(llm, meta: dict, question: str) -> list[str]:
    """For large schemas, ask the model which tables the question needs."""
    if len(meta) <= TABLE_SELECT_THRESHOLD:
        return list(meta)
    overview = "\n".join(
        f"{t}: {', '.join(n for n, _ in i['columns'])}"
        + (f" -- {i['description']}" if i["description"] else "")
        for t, i in meta.items()
    )
    prompt = (
        f"Database tables and columns:\n{overview}\n\n"
        f"Question: {question}\n"
        "Return a JSON array with the names of the tables needed to answer it "
        "(include join/bridge tables). Return only the JSON array."
    )
    try:
        found = json.loads(re.search(r"\[.*\]", llm.invoke(prompt).content, re.S).group())
        chosen = {t for t in found if t in meta}
    except Exception:
        return list(meta)
    for t in list(chosen):  # keep tables reachable by foreign key so joins stay possible
        chosen.update(ref for _, ref, _ in meta[t]["fks"] if ref in meta)
    return sorted(chosen) or list(meta)


# ---------- SQL generation and execution ----------
def clean_sql(raw: str) -> str:
    raw = re.sub(r"^```(?:sql)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return raw.rstrip(";").strip()


def check_read_only(sql: str) -> str | None:
    """Return a reason the query is rejected, or None if it looks safe."""
    lowered = sql.lower()
    if not lowered.startswith(("select", "with", "show", "describe")):
        return "only SELECT/WITH/SHOW/DESCRIBE statements are allowed"
    if ";" in sql:
        return "only a single statement is allowed"
    if re.search(r"\b(insert|update|delete|drop|alter|truncate|create|grant|replace|into)\b", lowered):
        return "the query contains a write or export keyword"
    return None


def run_query(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        # The DB enforces read-only, so this holds even if the keyword check above is bypassed.
        conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
        conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={QUERY_TIMEOUT_MS}"))
        conn.commit()  # the SET applies from the next transaction
        return pd.read_sql(text(sql), conn)


def build_history() -> list:
    messages = []
    done = [m for m in st.session_state.messages if m["role"] == "assistant" and m.get("sql")]
    asked = [m for m in st.session_state.messages if m["role"] == "user"]
    for q, a in list(zip(asked, done))[-HISTORY_TURNS:]:
        messages += [HumanMessage(q["content"]), AIMessage(a["sql"])]
    return messages


def generate_and_run(llm, engine, schema: str, question: str):
    """Generate SQL, run it, and on failure feed the error back to the model."""
    system = (
        "You are a MySQL expert. Schema:\n"
        f"{schema}\n\n"
        "Write ONE read-only MySQL query that answers the user's question. "
        f"Add LIMIT {MAX_ROWS} unless it is an aggregate. "
        "Earlier questions in this chat may be referred to by follow-ups. "
        "Return only the SQL, no explanation."
    )
    messages = [SystemMessage(system), *build_history(), HumanMessage(question)]
    sql, error = "", ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        sql = clean_sql(llm.invoke(messages).content)
        error = check_read_only(sql)
        if error is None:
            try:
                return sql, run_query(engine, sql), attempt, None
            except Exception as e:
                error = str(getattr(e, "orig", e))
        messages += [
            AIMessage(sql),
            HumanMessage(f"That query was rejected or failed: {error}\nFix it. Return only the SQL."),
        ]
    return sql, None, MAX_ATTEMPTS, error


def summarize(llm, question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "I couldn't find any matching data for that question."
    shown = df.head(50)
    note = f" (showing first 50 of {len(df)} rows)" if len(df) > 50 else ""
    prompt = (
        f"Question: {question}\n"
        f"Query result{note}:\n{shown.to_string(index=False)}\n\n"
        "Write ONE short sentence that introduces this result as the answer, "
        "for example 'The top five actors by number of films are:'. "
        "The result is shown in a table right after your sentence, so do not list "
        "the rows, do not use bullets or tables, and do not mention SQL or how the "
        "data was fetched. If the result is a single value, state it in the sentence."
    )
    return intro_line(llm.invoke(prompt).content)


def intro_line(text_: str) -> str:
    """Keep only the lead-in sentence, dropping any rows the model echoed after it."""
    first = text_.strip().splitlines()[0] if text_.strip() else ""
    m = re.match(r"^(.*?(?::(?=\s|$)|[.!?](?=\s|$)))", first)
    return (m.group(1) if m else first).strip()


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).replace("_", " ").title())


# ---------- query log ----------
def log_query(question, sql, attempts, rows, error):
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "db_user": st.session_state.get("db_user", ""),
        "question": question,
        "sql": sql,
        "attempts": attempts,
        "rows": rows,
        "error": error or "",
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_log(limit=50) -> pd.DataFrame:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
        return pd.DataFrame([json.loads(line) for line in reversed(lines)])
    except (OSError, ValueError):
        return pd.DataFrame()


# ---------- sidebar ----------
with st.sidebar:
    st.header("MySQL connection")
    host = st.text_input("Host", os.getenv("MYSQL_HOST", "localhost"))
    port = st.text_input("Port", os.getenv("MYSQL_PORT", "3306"))
    user = st.text_input("User", os.getenv("MYSQL_USER", "root"))
    password = st.text_input("Password", os.getenv("MYSQL_PASSWORD", ""), type="password")
    database = st.text_input("Database", os.getenv("MYSQL_DATABASE", ""))

    if st.button("Connect", type="primary"):
        try:
            engine = get_engine(host, port, user, password, database)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            st.session_state.engine = engine
            st.session_state.meta = load_metadata(engine)
            st.session_state.sample_cache = {}
            st.session_state.db_user = user
            st.session_state.messages = []
            st.success(f"Connected ({len(st.session_state.meta)} tables)")
        except Exception as e:
            st.session_state.pop("engine", None)
            st.error(f"Connection failed: {e}")

    if "engine" in st.session_state and "meta" in st.session_state:
        st.subheader("Try asking")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex-{q}", width="stretch"):
                st.session_state.pending = q

        with st.expander("Tables"):
            st.write(", ".join(st.session_state.meta))
        with st.expander("Query log"):
            log_df = read_log()
            if log_df.empty:
                st.caption("No queries yet.")
            else:
                st.dataframe(log_df, width="stretch", hide_index=True)

# ---------- main: chat ----------
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing from .env")
    st.stop()

if "engine" not in st.session_state or "meta" not in st.session_state:
    st.info("Connect to your database from the sidebar to start.")
    st.stop()

llm = ChatGroq(model=MODEL, temperature=0)
engine = st.session_state.engine

for msg in st.session_state.setdefault("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("df") is not None:
            st.dataframe(pretty(msg["df"]), width="stretch", hide_index=True)

question = st.chat_input("e.g. How many customers signed up last month?")
question = question or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        reply = {"role": "assistant", "content": ""}
        sql, attempts, rows, error = "", 0, 0, None
        try:
            with st.spinner("Thinking..."):
                meta = st.session_state.meta
                tables = select_tables(llm, meta, question)
                schema = "\n\n".join(describe_table(engine, t, meta[t]) for t in tables)
                sql, df, attempts, error = generate_and_run(llm, engine, schema, question)
            reply["sql"] = sql  # kept for follow-up context and the log, not shown
            if df is None:
                reply["content"] = (
                    "Sorry, I couldn't work out an answer to that. "
                    "Try rephrasing the question or being more specific."
                )
            else:
                rows = len(df)
                with st.spinner("Preparing answer..."):
                    reply["content"] = summarize(llm, question, df)
                if not df.empty:
                    reply["df"] = df
        except Exception as e:
            error = str(e)
            reply["content"] = "Sorry, something went wrong while answering that. Please try again."
        st.markdown(reply["content"])
        if reply.get("df") is not None:
            st.dataframe(pretty(reply["df"]), width="stretch", hide_index=True)
    st.session_state.messages.append(reply)
    log_query(question, sql, attempts, rows, error)

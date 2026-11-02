# AskDB

Ask questions about your MySQL database in plain English. AskDB turns each question into a read-only SQL query, runs it, and shows a one-line explanation with the result as a table.

Built with Streamlit, LangChain, Groq and SQLAlchemy. Tested against the MySQL Sakila sample database.

## Features

- Chat interface with follow-up questions (remembers the last few turns)
- Self-correcting SQL: failed queries are sent back to the model for up to 3 retries
- Schema-aware prompts: columns, primary and foreign keys, and sample rows
- Large schemas: picks only the relevant tables before generating SQL
- Read-only by design: keyword checks, a read-only MySQL session and a query timeout
- Clickable example questions and a query log in the sidebar

## Setup

1. Install dependencies:

   ```
   pip install -r req.txt
   ```

2. Create a `.env` file:

   ```
   GROQ_API_KEY=your-groq-api-key
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=your-user
   MYSQL_PASSWORD=your-password
   MYSQL_DATABASE=sakila
   # optional: GROQ_MODEL=openai/gpt-oss-120b
   ```

3. Run the app:

   ```
   streamlit run main.py
   ```

Use a MySQL user with SELECT-only permission.

## Optional

Add a `table_descriptions.json` next to `main.py` to help the model understand your tables:

```json
{ "rental": "One row per DVD rental", "payment": "Customer payments" }
```

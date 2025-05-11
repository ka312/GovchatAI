# GovchatAI


A natural language interface for querying government contracts data stored in PostgreSQL databases. This application allows users to ask questions in plain English about government contracts and receive detailed insights without needing to write SQL or understand the underlying database schema.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [How It Works](#how-it-works)
- [Example Queries](#example-queries)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Dependencies](#dependencies)

## Overview

GovSearch AI transforms complex government contract data into accessible insights through a conversational interface. Users can ask natural language questions about government contracts, and the system will automatically generate appropriate SQL queries, execute them against the database, and present the results in a user-friendly format.

## Features

- **Natural Language Querying:** Ask questions in plain English about government contracts
- **Conversational Memory:** Maintains context across multiple queries for follow-up questions
- **Automatic SQL Generation:** Converts natural language to optimized PostgreSQL queries
- **Entity Tracking:** Tracks mentioned entities across conversation for enhanced context
- **Interactive Results:** View query results directly in the Streamlit interface
- **Query History:** Review previous queries and responses
- **Export Capability:** Download large result sets as CSV files
- **Error Handling:** Graceful handling of query errors with user-friendly messages

## Technical Architecture

- **Frontend:** Streamlit web interface
- **Backend:** Python application with PostgreSQL connection
- **AI Components:** Azure OpenAI integration via LangChain
- **Memory Management:** ConversationBufferMemory for maintaining chat history
- **Query Tracking:** Custom QueryTracker class for maintaining context between queries

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/govsearch-ai.git
   cd govsearch-ai
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install streamlit pandas psycopg2-binary python-dotenv langchain-core langchain-openai
   ```

## Setup

1. Create a `.env` file in the project root with the following variables:
   ```
   # Database Configuration
   DB_NAME=your_database_name
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_API_VERSION=your_api_version
   ```

2. Ensure your PostgreSQL database has the `tm_awards` table structured according to the column definitions in the code.

## Running the Application

Start the Streamlit app:
streamlit run chat.py
The application will be accessible at `http://localhost:8501` by default.

## How It Works

### 1. Query Processing Flow

1. **User Input:** User enters a natural language question in the Streamlit interface
2. **SQL Generation:** 
   - The `generate_sql_query()` function passes the question to Azure OpenAI
   - It provides context from schema definitions, conversation history, and previous queries
   - The LLM generates a SQL query tailored to the PostgreSQL database
3. **Query Execution:** 
   - The `execute_sql_query()` function connects to PostgreSQL and runs the generated query
   - Results are returned as a pandas DataFrame
4. **Answer Generation:**
   - The `refine_answer()` function sends the query results back to the LLM
   - It formats the data in a human-readable way based on query type and result count
5. **Response Display:**
   - The interface shows the generated SQL, the natural language answer, and results table
   - For large result sets, a download option is provided

### 2. Key Components

#### QueryTracker
Maintains context across multiple queries by tracking:
- Previously executed SQL queries
- Result counts from previous queries
- WHERE clauses that can be reused
- Entity mentions and their counts

#### Entity Analysis
The `analyze_previous_response()` function extracts mentions of entities (like "contracts" or "awards") and their counts from AI responses using regex patterns.

#### Conversation Memory
Uses LangChain's `ConversationBufferMemory` to maintain a history of the conversation, enabling the model to reference previous interactions.

#### SQL Generation
Provides detailed context to the LLM including:
- Database schema information
- Sample data
- Previous queries and results
- Entity mentions
- Recent conversation history

#### Answer Refinement
Processes query results through a specialized prompt that instructs the LLM to:
- Format data in a user-friendly way
- Detect specific query types (e.g., contract details requests)
- Include appropriate footer text based on result count
- Format numbers with commas and currency symbols

## Example Queries

- "Show me all contracts worth more than $1 million from the Department of Defense"
- "Which contractors received the most awards last year?"
- "List active task orders related to IT services"
- "How many contracts were awarded to small businesses?"
- "What is the total value of contracts in California?"
- "Show details of the largest contract by value"
- "Who are the top 5 contractors by total obligation amount?"

## Project Structure

- `chat.py`: Main application file containing all components
- `.env`: Environment variables for database and Azure OpenAI configuration
- `README.md`: Project documentation

Key sections in `chat.py`:
- PostgreSQL Configuration: Sets up database connection parameters
- Table Schema: Defines column definitions for the database
- Query Tracking System: Implements context tracking between queries
- Database and Query Functions: Handles SQL generation and execution
- Streamlit Interface: Manages the web UI and user interactions

## Customization

### Adapting to Different Databases

To adapt this application to a different database schema:
1. Update the `TABLE_NAME` variable
2. Replace the `COLUMN_DEFINITIONS` dictionary with your schema
3. Modify the example queries and prompts in the Streamlit interface

### Changing AI Provider

While this implementation uses Azure OpenAI, you can modify the LLM configuration to use other providers:
1. Replace the `AzureChatOpenAI` import and initialization
2. Update environment variables accordingly

## Dependencies

- `streamlit`: Web interface
- `pandas`: Data handling and processing
- `psycopg2-binary`: PostgreSQL database connection
- `python-dotenv`: Environment variables management
- `langchain-core`: Core LangChain functionality
- `langchain-openai`: OpenAI integration for LangChain
- `re`: Regular expression processing for entity extraction
- `json`: JSON handling for schema definitions
- `datetime`: Timestamp generation for CSV exports

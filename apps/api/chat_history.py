# Enable annotations for better type hinting support
from __future__ import annotations

# Import standard libraries for database access and time
import os
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger('chat_history')

# Define database connection parameters from environment
DATABASE_URL = os.getenv("DATABASE_URL", 'postgresql://postgres:admin@postgres:5432/postgres')

# Class to manage chat history in PostgreSQL
class ChatHistory:
    def __init__(self):
        # Initialize database connection
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)  # Use dict cursor for row results

    def create_session(self, session_id: str) -> None:
        # Create a new chat session with timestamp
        query = """
        INSERT INTO chat_sessions (session_id, created_at)
        VALUES (%s, %s)
        ON CONFLICT (session_id) DO NOTHING
        """
        self.cursor.execute(query, (session_id, int(time.time())))
        self.conn.commit()
        log.info(f"Created chat session: {session_id}")

    def add_message(self, session_id: str, question: str, answer: str) -> None:
        # Add a question-answer pair to the session
        query = """
        INSERT INTO chat_messages (session_id, question, answer, created_at)
        VALUES (%s, %s, %s, %s)
        """
        self.cursor.execute(query, (session_id, question, answer, int(time.time())))
        self.conn.commit()
        log.info(f"Added message to session {session_id}")

    def get_session_history(self, session_id: str, limit: int = 15) -> List[Dict[str, Any]]:
        # Retrieve the last 'limit' messages for a session
        query = """
        SELECT question, answer, created_at
        FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        self.cursor.execute(query, (session_id, limit))
        return self.cursor.fetchall()

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        # Retrieve all session IDs with their creation times
        query = """
        SELECT session_id, created_at
        FROM chat_sessions
        ORDER BY created_at DESC
        LIMIT 10
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def prune_old_sessions(self) -> None:
        # Keep only the 10 most recent sessions, delete older ones
        query = """
        WITH recent_sessions AS (
            SELECT session_id
            FROM chat_sessions
            ORDER BY created_at DESC
            LIMIT 10
        )
        DELETE FROM chat_sessions
        WHERE session_id NOT IN (SELECT session_id FROM recent_sessions)
        """
        self.cursor.execute(query)
        self.conn.commit()
        # Also delete messages for deleted sessions
        query = """
        DELETE FROM chat_messages
        WHERE session_id NOT IN (SELECT session_id FROM chat_sessions)
        """
        self.cursor.execute(query)
        self.conn.commit()
        log.info("Pruned old chat sessions")

    def close(self) -> None:
        # Close database connection
        self.cursor.close()
        self.conn.close()
        log.info("Closed chat history database connection")
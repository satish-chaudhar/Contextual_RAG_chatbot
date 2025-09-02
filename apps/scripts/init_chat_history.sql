-- Create table for chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id VARCHAR(36) PRIMARY KEY,  -- Unique session ID (UUID)
    created_at INTEGER NOT NULL  -- Unix timestamp for session creation
);

-- Create table for chat messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,  -- Auto-incrementing message ID
    session_id VARCHAR(36) REFERENCES chat_sessions(session_id),  -- Foreign key to session
    question TEXT NOT NULL,  -- User question
    answer TEXT NOT NULL,  -- Assistant response
    created_at INTEGER NOT NULL  -- Unix timestamp for message
);

-- Create index for faster queries on session_id
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
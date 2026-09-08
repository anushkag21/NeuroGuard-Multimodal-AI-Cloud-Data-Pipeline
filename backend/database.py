"""
NeuroGuard Clinic – Database Module
MySQL storage for sessions with full structured data.
Falls back to in-memory storage if MySQL is unavailable.
"""
import os
import json
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class Database:
    """MySQL database operations for NeuroGuard Clinic."""

    def __init__(self):
        self.config = {
    'host': os.getenv('MYSQL_HOST'),
    'port': int(os.getenv('MYSQL_PORT', '3306')),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE', 'neuroguard'),
}
        
    
        self.available = False
        self._memory_sessions = []  # fallback
        self.init_db()

    def get_connection(self):
        import mysql.connector
        return mysql.connector.connect(**self.config)

    def init_db(self):
        """Initialize database tables."""
        try:
            import mysql.connector
            # First ensure the database exists
            cfg_no_db = {k: v for k, v in self.config.items() if k != 'database'}
            conn = mysql.connector.connect(**cfg_no_db)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.config['database']}`")
            conn.commit()
            cursor.close()
            conn.close()

            # Now connect to the database and create tables
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(50) UNIQUE NOT NULL,
                    patient_id VARCHAR(50) NOT NULL,
                    patient_name VARCHAR(100) DEFAULT 'Unknown',
                    score INT DEFAULT 0,
                    severity VARCHAR(30) DEFAULT 'Normal',
                    confidence FLOAT DEFAULT 0.0,
                    mode VARCHAR(30) DEFAULT 'full_multimodal',
                    sensors_used TEXT,
                    duration_seconds INT DEFAULT 0,
                    answers_json TEXT,
                    sensor_summary_json TEXT,
                    report_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Add patient_name column retroactively if it's missing from previous runs
            try:
                cursor.execute('ALTER TABLE sessions ADD COLUMN patient_name VARCHAR(100) DEFAULT "Unknown"')
            except Exception:
                pass
            conn.commit()
            cursor.close()
            conn.close()
            self.available = True
            print("[Database] MySQL initialized successfully")
        except Exception as e:
            self.available = False
            print(f"[Database] MySQL unavailable, using in-memory fallback: {e}")

    def save_session(self, session_data):
        """Save a completed test session."""
        if not self.available:
            session_data['id'] = len(self._memory_sessions) + 1
            session_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._memory_sessions.insert(0, session_data)
            return session_data.get('session_id')

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions
                    (session_id, patient_id, patient_name, score, severity, confidence, mode,
                     sensors_used, duration_seconds, answers_json, sensor_summary_json, report_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                session_data.get('session_id', ''),
                session_data.get('patient_id', ''),
                session_data.get('patient_name', 'Unknown'),
                session_data.get('score', 0),
                session_data.get('severity', 'Normal'),
                session_data.get('confidence', 0.0),
                session_data.get('mode', 'unknown'),
                json.dumps(session_data.get('sensors_used', [])),
                session_data.get('duration_seconds', 0),
                json.dumps(session_data.get('answers', [])),
                json.dumps(session_data.get('sensor_summary', {})),
                json.dumps(session_data.get('report', {})),
            ))
            conn.commit()
            cursor.close()
            conn.close()
            return session_data.get('session_id')
        except Exception as e:
            print(f"[Database] Error saving session: {e}")
            # Fallback to memory
            session_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._memory_sessions.insert(0, session_data)
            return session_data.get('session_id')

    def get_session(self, session_id):
        """Retrieve a session by session_id."""
        if not self.available:
            for s in self._memory_sessions:
                if s.get('session_id') == session_id:
                    return s
            return None

        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT * FROM sessions WHERE session_id = %s', (session_id,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if row:
                # Parse JSON fields
                row['sensors_used'] = json.loads(row.get('sensors_used', '[]') or '[]')
                row['answers'] = json.loads(row.get('answers_json', '[]') or '[]')
                row['sensor_summary'] = json.loads(row.get('sensor_summary_json', '{}') or '{}')
                row['report'] = json.loads(row.get('report_json', '{}') or '{}')
            return row
        except Exception as e:
            print(f"[Database] Error getting session: {e}")
            # Try memory fallback
            for s in self._memory_sessions:
                if s.get('session_id') == session_id:
                    return s
            return None

    def get_all_sessions(self):
        """Retrieve all past sessions, newest first."""
        if not self.available:
            return list(self._memory_sessions)

        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT * FROM sessions ORDER BY created_at DESC LIMIT 100')
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            for row in rows:
                row['sensors_used'] = json.loads(row.get('sensors_used', '[]') or '[]')
            return rows
        except Exception as e:
            print(f"[Database] Error getting sessions: {e}")
            return list(self._memory_sessions)

    def delete_session(self, session_id):
        """Delete a session by session_id."""
        if not self.available:
            self._memory_sessions = [s for s in self._memory_sessions if s.get('session_id') != session_id]
            return True

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE session_id = %s', (session_id,))
            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"[Database] Error deleting session: {e}")
            return False

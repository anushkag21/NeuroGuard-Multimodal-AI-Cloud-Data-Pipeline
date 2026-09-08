import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
)

cursor = conn.cursor()

cursor.execute("SELECT DATABASE(), VERSION();")
result = cursor.fetchone()

print("✓ Azure MySQL connection successful!")
print("Database:", result[0])
print("MySQL Version:", result[1])

cursor.close()
conn.close()
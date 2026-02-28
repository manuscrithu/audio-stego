import os

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


load_dotenv()


# DATABASE_URL = os.getenv("DATABASE_URL")
# DB_HOST = os.getenv("DB_HOST", "centerbeam.proxy.rlwy.net")
# DB_PORT = int(os.getenv("DB_PORT", "13454"))
# DB_NAME = os.getenv("DB_NAME", "railway")
# DB_USER = os.getenv("DB_USER", "postgres")
# DB_PASSWORD = os.getenv("DB_PASSWORD", "<POSTGRES_PASSWORD>")
# DB_SSLMODE = os.getenv("DB_SSLMODE", "require")


DATABASE_URL = "postgresql://songdata_45q5_user:YO6ox9GwBYYvZeBavNbTKRs0p7mCi1PU@dpg-d6h9183uibrs739r35vg-a/songdata_45q5"
DB_HOST = "d6h9183uibrs739r35vg-a"
DB_PORT = "5432"
DB_NAME = "songdata_45q5"
DB_USER = "songdata_45q5_user"
DB_PASSWORD = "YO6ox9GwBYYvZeBavNbTKRs0p7mCi1PU"
DB_SSLMODE = "require"

def get_connection():
    if DATABASE_URL:
        return psycopg2.connect(
            DATABASE_URL,
            sslmode=DB_SSLMODE,
            cursor_factory=RealDictCursor,
        )

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
        cursor_factory=RealDictCursor,
    )


def init_db():
    query = '''
    CREATE TABLE IF NOT EXISTS "song-data" (
        id BIGSERIAL PRIMARY KEY,
        "songName" TEXT NOT NULL,
        score DOUBLE PRECISION NOT NULL
    );
    '''

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()


def save_song_data(song_name: str, score: float):
    query = '''
    INSERT INTO "song-data" ("songName", score)
    VALUES (%s, %s)
    RETURNING id;
    '''

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (song_name, score))
            row = cur.fetchone()
        conn.commit()

    return row["id"] if row else None

"""
Module for establishing and testing an asynchronous database connection using SQLAlchemy.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy.sql import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable is not set.")

engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def test_connection() -> None:
    """
    Test the database connection by executing a simple SQL statement.

    This function attempts to begin a connection using the asynchronous engine & executes a raw SQL 
    query ("SELECT 1") to verify database connectivity. It prints the result of the test.
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            print("Database connected successfully!")
    except Exception as e: # pylint: disable=broad-except
        print(f"Database connection failed: {e}")

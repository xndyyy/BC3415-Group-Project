import os
import sqlite3
import uuid
import re
import asyncio
from typing import List, Tuple

import aiohttp
from more_itertools import chunked
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.asyncio import tqdm_asyncio

from vectorizer.app.core.settings import get_settings
from vectorizer.app.core.logger import logger
from .chunkenizer import recursive_character_splitting
from vectorizer.app.embeddings.embedding_generator import generate_embedding  # local, 384-dim

settings = get_settings()


class VectorDB:
    """
    Qdrant vector store helper using a local SentenceTransformer encoder (all-MiniLM-L6-v2).
    - Vector size: 384
    - Distance: COSINE
    """

    def __init__(self, table_name: str, collection_name: str, create_collection: bool = False):
        self.table_name = table_name
        self.collection_name = collection_name
        self.connect_to_qdrant()
        if create_collection:
            self.create_or_clear_collection()

    # ---------------------------
    # Qdrant lifecycle
    # ---------------------------
    def connect_to_qdrant(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        logger.info("Connected to Qdrant")

    def create_or_clear_collection(self):
        if self.client.collection_exists(self.collection_name):
            logger.info(f"Collection {self.collection_name} already exists. Recreating it.")
            self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logger.info(f"Created collection: {self.collection_name}")

    # ---------------------------
    # Content formatting
    # ---------------------------
    def format_content(self, data, collection_name: str) -> str:
        if collection_name == "car_rentals_collection":
            booking_status = "booked" if data["booked"] else "not booked"
            return (
                f"Car rental: {data['name']}, located at: {data['location']}, price tier: {data['price_tier']}. "
                f"Rental period starts on {data['start_date']} and ends on {data['end_date']}. "
                f"Currently, the rental is: {booking_status}."
            )

        elif collection_name == "excursions_collection":
            booking_status = "booked" if data["booked"] else "not booked"
            return (
                f"Excursion: {data['name']} at {data['location']}. "
                f"Additional details: {data['details']}. "
                f"Currently, the excursion is {booking_status}. "
                f"Keywords: {data['keywords']}."
            )

        elif collection_name == "flights_collection":
            return (
                f"Flight {data['flight_no']} from {data['departure_airport']} to {data['arrival_airport']} "
                f"was scheduled to depart at {data['scheduled_departure']} and arrive at {data['scheduled_arrival']}. "
                f"The actual departure was at {data['actual_departure']} and the actual arrival was at {data['actual_arrival']}. "
                f"Currently, the flight status is '{data['status']}' and it was operated with aircraft code {data['aircraft_code']}."
            )

        elif collection_name == "hotels_collection":
            booking_status = "booked" if data["booked"] else "not booked"
            return (
                f"Hotel {data['name']} located in {data['location']} is categorized as {data['price_tier']} tier. "
                f"The check-in date is {data['checkin_date']} and the check-out date is {data['checkout_date']}. "
                f"Currently, the booked status is: {booking_status}."
            )

        elif collection_name == "faq_collection":
            return data["page_content"]  # Return the page content directly for FAQ

        return str(data)

    # ---------------------------
    # Local embedding (async wrapper)
    # ---------------------------
    async def generate_embedding_async(self, content: str, _session=None) -> List[float]:
        """
        Async wrapper around the local SentenceTransformer encoder (sync).
        Runs in default thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, generate_embedding, content)

    async def process_chunk(self, chunk: str, metadata: dict, _session=None) -> PointStruct:
        embedding = await self.generate_embedding_async(chunk)
        return PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"content": chunk, **metadata},
        )

    # ---------------------------
    # Indexing entrypoints
    # ---------------------------
    async def create_embeddings_async(self):
        if self.table_name == "faq":
            await self.index_faq_docs()
        else:
            await self.index_regular_docs()

    def create_embeddings(self):
        asyncio.run(self.create_embeddings_async())

    # ---------------------------
    # Regular tables → formatted text → chunks → embeddings
    # ---------------------------
    async def index_regular_docs(self):
        # Load rows from SQLite
        db_connection = sqlite3.connect(settings.SQLITE_DB_PATH)
        cursor = db_connection.cursor()
        cursor.execute(f"SELECT * FROM {self.table_name}")
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        db_connection.close()

        if not rows:
            logger.warning(f"No data found in table {self.table_name}")
            return

        data = [dict(zip(column_names, row)) for row in rows]

        # Build (chunk, metadata) pairs with correct alignment
        paired: List[Tuple[str, dict]] = []
        for item in data:
            text = self.format_content(item, self.collection_name)
            for chunk in recursive_character_splitting(text):
                if chunk:
                    paired.append((chunk, item))

        if not paired:
            logger.warning(f"No valid chunks generated for {self.collection_name}")
            return

        batch_size = 100
        total_points = 0

        for bstart in range(0, len(paired), batch_size):
            batch = paired[bstart : bstart + batch_size]

            tasks = [self.process_chunk(chunk, meta) for (chunk, meta) in batch]

            points = []
            for coro in tqdm_asyncio.as_completed(
                tasks,
                desc=f"Generating embeddings for {self.collection_name} (batch {bstart // batch_size + 1})",
                total=len(tasks),
            ):
                try:
                    pt = await coro
                    if pt is not None:
                        points.append(pt)
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")

            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
                total_points += len(points)
                logger.info(
                    f"Indexed {len(points)} documents into {self.collection_name} "
                    f"(batch {bstart // batch_size + 1})"
                )

        logger.info(
            f"Finished indexing. Total documents indexed into {self.collection_name}: {total_points}"
        )

    # ---------------------------
    # FAQ: download markdown, split, embed locally
    # ---------------------------
    async def index_faq_docs(self):
        faq_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"

        async with aiohttp.ClientSession() as session:
            async with session.get(faq_url) as response:
                faq_text = await response.text()

        docs = [{"page_content": txt.strip()} for txt in re.split(r"(?=\n##)", faq_text) if txt.strip()]

        # Create points
        tasks = [self.process_chunk(doc["page_content"], {"type": "faq"}) for doc in docs]
        points = await tqdm_asyncio.gather(*tasks, desc="Generating embeddings for FAQ documents")

        # Upsert in batches
        inserted = 0
        for batch in chunked(points, 100):
            self.client.upsert(collection_name=self.collection_name, points=list(batch))
            inserted += len(list(batch))

        if inserted:
            logger.info(f"Indexed {inserted} FAQ documents into {self.collection_name}.")
        else:
            logger.warning("No FAQ documents were successfully embedded and indexed.")

    # ---------------------------
    # Search
    # ---------------------------
    def search(self, query: str, limit: int = 2, with_payload: bool = True):
        """
        Synchronous search helper: embeds the query locally then queries Qdrant.
        """
        query_vector = generate_embedding(query)  # local, 384-dim
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=with_payload,
        )


if __name__ == "__main__":
    vectordb = VectorDB("example_table", "example_collection", create_collection=True)
    vectordb.create_embeddings()

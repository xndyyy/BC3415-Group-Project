from customer_support_chat.app.core import logger
from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import sqlite3
from typing import Optional, Union, List, Dict
from openai import OpenAI
import json, datetime
import re

settings = get_settings()
db = settings.SQLITE_DB_PATH
flights_vectordb = VectorDB(table_name="flights", collection_name="flights_collection")
client = OpenAI()

@tool
def fetch_user_flight_information(*, config: RunnableConfig) -> List[Dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        LEFT JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results

def ai_extract_flight_intent(query: str):
    """
    Ask an LLM to infer {origin, destination, start, end}
    directly from the natural-language query.
    """
    prompt = f"""
    You are a travel-booking parser. 
    Read the user query below and return a JSON object with:
      - "origin": city of departure (string or null)
      - "destination": city of arrival (string or null)
      - "start": ISO 8601 date string (YYYY-MM-DD) for departure (or null)
      - "end": ISO 8601 date string for arrival/return window (1 day after start, or null)
    
    User query: {query}
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    parsed = json.loads(completion.choices[0].message.content)

    # Auto-compute end date if missing but start present
    if parsed.get("start") and not parsed.get("end"):
        dt = datetime.datetime.fromisoformat(parsed["start"])
        parsed["end"] = (dt + datetime.timedelta(days=1)).date().isoformat()

    return parsed

@tool
def search_flights(query: str, limit: int = 5, *, config: RunnableConfig = None):
    """
    Context-aware flight search tool.
    Keeps track of missing info and prevents infinite loops.
    """

    config = config or {}
    context = config.setdefault("configurable", {})
    flight_ctx = context.setdefault("flight_context", {})

    # Parse user intent
    parsed = ai_extract_flight_intent(query)

    # Persist values across turns
    for key in ["origin", "destination", "start", "end"]:
        if parsed.get(key):
            flight_ctx[key] = parsed[key]
        elif key not in flight_ctx:
            flight_ctx[key] = None

    # Ensure end date if start is provided
    if flight_ctx.get("start") and not flight_ctx.get("end"):
        flight_ctx["end"] = flight_ctx["start"] + datetime.timedelta(days=1)

    # Debug print to confirm persistence
    # print("DEBUG merged flight_ctx:", flight_ctx)

    # Determine missing fields
    missing = [
        k for k in ["origin", "destination", "start"]
        if not flight_ctx.get(k)
    ]

    if missing:
        context["last_prompt"] = missing[0]
        if missing[0] == "origin":
            return "Got it. From which city will you be departing?"
        elif missing[0] == "destination":
            return "Sure! Where would you like to fly to?"
        elif missing[0] == "start":
            return "When would you like to travel? (e.g., this weekend or a specific date)"
        return

    # Run SQL query
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    sql = """
    SELECT 
        f.flight_no,
        f.departure_airport,
        dep.city AS departure_city,
        f.arrival_airport,
        arr.city AS arrival_city,
        f.scheduled_departure,
        f.scheduled_arrival,
        f.status,
        f.aircraft_code
    FROM flights f
    JOIN airports_data dep ON f.departure_airport = dep.airport_code
    JOIN airports_data arr ON f.arrival_airport = arr.airport_code
    WHERE LOWER(dep.city) LIKE '%' || LOWER(?) || '%'
        AND LOWER(arr.city) LIKE '%' || LOWER(?) || '%'
        AND substr(f.scheduled_departure,1,10) BETWEEN ? AND ?
    ORDER BY f.scheduled_departure ASC
    LIMIT ?
    """

    params = [
        f"%{parsed['origin'].lower()}%",
        f"%{parsed['destination'].lower()}%",
        parsed['start'],
        parsed['end'],
        limit,
    ]

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    columns = [c[0] for c in cursor.description]
    results = [dict(zip(columns, row)) for row in rows]
    conn.close()

    if not results:
        return [{"message": "No matching flights found. Try changing your date or city."}]

    context["flight_context"] = {}
    context["last_prompt"] = None

    return results

@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # Check if the ticket exists and belongs to the passenger
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    ticket = cursor.fetchone()
    if not ticket:
        conn.close()
        return f"Ticket {ticket_no} not found for passenger {passenger_id}."

    # Update the flight in ticket_flights
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Ticket {ticket_no} successfully updated to flight {new_flight_id}."
    else:
        conn.close()
        return f"Failed to update ticket {ticket_no}."

@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # Check if the ticket exists and belongs to the passenger
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    ticket = cursor.fetchone()
    if not ticket:
        conn.close()
        return f"Ticket {ticket_no} not found for passenger {passenger_id}."

    # Delete from ticket_flights
    cursor.execute(
        "DELETE FROM ticket_flights WHERE ticket_no = ?",
        (ticket_no,),
    )
    # Delete from tickets
    cursor.execute(
        "DELETE FROM tickets WHERE ticket_no = ?",
        (ticket_no,),
    )
    conn.commit()

    conn.close()
    return f"Ticket {ticket_no} successfully cancelled."

# Swiss Airlines AI Travel Agent

A multi-agent, retrieval-augmented customer support system that helps users manage flights, book hotels, car rentals, and excursions, and get travel recommendations.  
Built with LangGraph, LangChain, Qdrant, and OpenAI, this project demonstrates how intelligent agents can collaborate across multiple domains while preserving user safety through approval logic.

---

## Features

- Retrieve a passenger’s booked flights
- Search and book:
  - Flights
  - Hotels
  - Car rentals
  - Excursions
- Airline policy lookup
- RAG-powered retrieval using Qdrant
- User approval flow for sensitive actions (e.g., flight cancellation)
- Accessible via command line interface (CLI) and Telegram bot

---

## System Architecture
```
User
│
▼
Primary Assistant ───> Policies / Hotel / Flights / Excursions / VectorDB
│
├─ Flight Booking Assistant
│ └─ cancel / change flights → requires approval
│
├─ Hotel Booking Assistant
│ └─ search + book
│
├─ Car Rental Assistant
│ └─ search + book
│
└─ Excursion Assistant
└─ recommend + book
```

Each assistant operates within the LangGraph workflow and shares state across interactions.

---

## Requirements

- Python 3.10+
- Poetry
- Docker (for Qdrant)
- OpenAI API key
- (Optional) Telegram bot token

---

## Local Setup

1. Clone the Repository:
```bash
git clone <repo-url>
cd <project-directory>
```

2. Create a `.env` file from `.dev.env`:
```bash
cp .dev.env .env
```


3. Edit the `.env` file and fill in the required values:

```bash
OPENAI_API_KEY="your_openai_api_key"
TELEGRAM_TOKEN="your_telegram_bot_token"
```


4. Install dependencies:

```bash
poetry install
```


5. Generate embeddings:

```bash
poetry run python vectorizer/app/main.py
```


6. Start the Qdrant vector database:

```bash
docker compose up qdrant -d
```


   You can access the Qdrant UI at: http://localhost:6333/dashboard#

7. Run the customer support chat system:

```bash
poetry run python ./ai_travel_agent/app/main.py
```
# ECKO

**ECKO** is a modular local AI companion system that combines:

* **LLM chat**
* **Streaming text-to-speech**
* **Speech-to-text**
* **Character personalities**
* **Persistent memory**
* **RAG knowledge retrieval**
* **Web interface**

The goal of ECKO is to provide a **fast, self-hosted conversational AI experience** that runs entirely on your own machine while remaining highly customizable.

It started as a simple streaming TTS frontend and has evolved into a **complete conversational stack** with character systems, memory, and knowledge retrieval.

* Ecko is designed to work first and foremost with Echo TTS and KoboldCPP. I've been impressed with MistralAI's LLM API performance and other providers (LLM and TTS) are also being added for fallback and user choice. Streaming audio may not be available under these configurations without paid subscription or even availaible at all. Ecko's intended low latency experience will take a hit but its perfectly usable. 

---

# Core Features

## Streaming Text-to-Speech

ECKO integrates with **Echo-TTS** for low-latency streaming voice playback.

* Real-time streaming speech
* Character voice presets
* Volume and gain control
* Optional audio processing
* IR reverb support

Unlike most TTS frontends, audio playback begins **immediately while generation is still happening**.

---

## LLM Chat Integration

ECKO connects to a local LLM server such as:

* **KoboldCPP**
* any compatible API endpoint

Features include:

* Character-based system prompts
* Context management
* Auto-continue generation
* Conversation persistence
* Configurable generation settings

---

## Character System

ECKO supports **per-character configuration**, allowing each personality to define:

* System prompts
* Voice presets
* Memory behavior
* Audio settings
* Personality metadata
* Avatar images

Characters are stored locally and can be easily extended.

---

## Speech-to-Text

Voice interaction is supported through integrated STT.

Features:

* Push-to-talk input
* Live transcription
* Mic toggle support
* Designed for local inference engines

---

## Persistent Memory

ECKO includes a **long-term memory system**.

The assistant can store and retrieve information between conversations.

Memory types may include:

* User facts
* Character notes
* Persistent conversation knowledge

All memory is stored locally.

---

## RAG Knowledge System

ECKO includes a **Retrieval Augmented Generation (RAG)** system.

This allows the assistant to reference external knowledge sources.

Knowledge sources can include:

* Conversation logs
* External text files
* Custom documents

RAG files are organized into:

```
rag/
 ├─ conversations/
 └─ extra/
```

---

## Web Interface

The application runs a **local web interface**.

This allows:

* Chat interaction
* Character selection
* Voice playback
* System configuration

The web server is launched from:

```
python ecko_web.py
```

Default port:

```
https://localhost:5050
```

---

## Web Search Integration

ECKO can optionally perform web searches to supplement responses when needed.

This allows the assistant to pull in **current information** outside the local knowledge base.

---

## Safety Layer

A configurable safety system helps control:

* Disallowed content
* Prompt filtering
* Output moderation

The system is fully local and customizable.

---

# Requirements

* **Python 3.11**
* Local **Echo-TTS API**
* Local **LLM server** (KoboldCPP recommended)

---

# Installation

## 1 Clone the repository

```
git clone https://github.com/ItsGeneralButtNaked/ecko
cd ecko
```

---

## 2 Create a virtual environment

### Conda

```
conda create -n ecko python=3.11
conda activate ecko
```

### Standard venv

```
python3.11 -m venv venv
source venv/bin/activate
```

---

## 3 Install dependencies

```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

# Running ECKO

Start the server:

```
python ecko_web.py
```

Then open your browser:

```
https://localhost:5050
```

The interface will load the chat system.

---

# Required External Services

ECKO expects the following services to be running.

## Echo-TTS API

Streaming text-to-speech server:

https://github.com/KevinAHM/echo-tts-api

or compatible implementations.

---

## LLM Server

Recommended:

```
KoboldCPP
```

https://github.com/LostRuins/koboldcpp

---

# Configuration

Most runtime settings are automatically loaded from:

```
ecko_session.json
```

This stores:

* current character
* UI state
* conversation settings
* generation parameters

---

# Platform Support

| Platform | Status                |
| -------- | --------------------- |
| Linux    | Fully supported       |
| Windows  | Works but less tested |
| MacOS    | Not officially tested |

---

# Design Goals

ECKO focuses on:

* **Local-first AI**
* **Low latency**
* **Character-driven interaction**
* **Simple extensibility**
* **Minimal external dependencies**

The system is intentionally modular so each component can evolve independently.

---

# Future Development

Possible future improvements include:

* bring the native desktop app back

---

# License

See repository license file.

---

# Acknowledgements

This project builds upon the work of several open-source communities:

* Echo-TTS
* KoboldCPP
* Local LLM ecosystem

Without them this project would not exist.

---

# Author

Created by **ItsGeneralButtNaked** as an experimental local AI interface.


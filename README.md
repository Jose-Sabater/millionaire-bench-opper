# Millionaire Bench Opper
Run [Millionaire Bench](https://github.com/ikiruneo/millionaire-bench/tree/main) concurrently and with many os and proprietary models


Transformed the input data into a better format and also translated it. [fragen_antwrten_en](fragen_antworten_en.json)
Each group of 15 questions is a "program". How many programs can each model complete successfully?
Earnings are calculated in the same way as the original show and the original millionaire bench.

# Leaderboard

*Last updated: September 3, 2025*

| Rank | Model | Avg Earnings (€) | Total Earnings (€) | Programs Won | Success Rate | Max Single Program (€) |
|------|-------|------------------|--------------------|--------------|--------------|-----------------------|
| 🥇 | xai/grok-4 | 771,494 | 34,717,250 | 34/45 | 75.6% | 1,000,000 |
| 🥈 | openai/gpt-5 | 728,922 | 32,801,500 | 32/45 | 71.1% | 1,000,000 |
| 🥉 | openai/gpt-5-mini | 576,506 | 25,942,750 | 24/45 | 53.3% | 1,000,000 |
| 4 | gcp/gemini-2.5-flash | 520,001 | 23,400,050 | 23/45 | 51.1% | 1,000,000 |
| 5 | fireworks/glm-4.5 | 407,521 | 18,338,450 | 17/45 | 37.8% | 1,000,000 |
| 6 | anthropic/claude-sonnet-4 | 370,954 | 16,692,950 | 15/45 | 33.3% | 1,000,000 |
| 7 | mistral/mistral-medium-2508-eu | 245,930 | 11,066,850 | 10/45 | 22.2% | 1,000,000 |
| 8 | openai/gpt-5-nano | 235,467 | 10,596,000 | 10/45 | 22.2% | 1,000,000 |
| 9 | berget/gpt-oss-120b | 205,642 | 9,253,900 | 8/45 | 17.8% | 1,000,000 |
| 10 | gcp/gemini-2.5-flash-lite | 94,410 | 4,248,450 | 3/45 | 6.7% | 1,000,000 |
| 11 | groq/moonshotai/kimi-k2-instruct | 86,370 | 3,886,650 | 3/45 | 6.7% | 1,000,000 |
| 12 | groq/gpt-oss-20b | 53,200 | 2,394,000 | 2/45 | 4.4% | 1,000,000 |
| 13 | anthropic/claude-3.5-haiku | 30,512 | 1,373,050 | 1/45 | 2.2% | 1,000,000 |
| 14 | groq/gemma2-9b-it | 3,308 | 148,850 | 0/45 | 0.0% | 125,000 |
| 15 | groq/llama-3.1-8b-instant | 106 | 4,750 | 0/45 | 0.0% | 2,000 |

**Total models evaluated:** 15 | **Programs per model:** 45 | **Total questions asked:** 10,125


## Run your own tests
1. Install dependencies using uv:
   ```bash
   uv sync
   ```

2. Run the benchmark:
   ```bash
   uv run millionaire-run.py
   ```

3. Rebuild leaderboard from individual results:
   ```bash
   python rebuild_leaderboard.py
   ```

# Resources
All data used comes from [Millionaire Bench](https://github.com/ikiruneo/millionaire-bench/tree/main), which uses data from https://github.com/GerritKainz/wer_wird_millionaer.

# Why this repository?
- **Concurrent execution**: Run all "programs" concurrently with very fast inference
- **Translation support**: Translate the original data so English readers understand the questions (questions are still asked in German to the models)
- **Convenience**: Uses [Opper](https://opper.ai) for ease of use, which provides:
  - Access to many open source and proprietary models ([see list](https://docs.opper.ai/capabilities/models))
  - Built in observability and retries
  - Convenient structured input/output for evaluating exact matches

# Example Translation
```json
{
"level": 1,
"question": "Ein bekanntes Sprichwort heisst: 'Alter... ?'",
"options": {
    "A": "Norwege",
    "B": "Schwede",
    "C": "Daene",
    "D": "Englaender"
},
"answer": "B",
"options_str": "A: Norwege, B: Schwede, C: Daene, D: Englaender",
"question_en": "A well-known proverb goes: 'Age... ?'",
"options_str_en": "A: Norwegian, B: Swede, C: Dane, D: Englishman"
},
{
"level": 2,
"question": "Worum handelt es sich, wenn man eine Ware direkt bei uebergabe des Pakets bezahlt?",
"options": {
    "A": "Vornahme",
    "B": "Nachnahme",
    "C": "Kosenahme",
    "D": "Taufnahme"
},
"answer": "B",
"options_str": "A: Vornahme, B: Nachnahme, C: Kosenahme, D: Taufnahme",
"question_en": "What is it called when you pay for merchandise directly upon delivery of the package?",
"options_str_en": "A: First name, B: Last name, C: Pet name, D: Baptismal name"
},
{
"level": 3,
"question": "Womit werden Eisenbahnwaggons professionell abgebremst?",
"options": {
    "A": "Riemchensandale",
    "B": "Lederstiefel",
    "C": "Badeschlapfen",
    "D": "Hemmschuh"
},
"answer": "D",
"options_str": "A: Riemchensandale, B: Lederstiefel, C: Badeschlapfen, D: Hemmschuh",
"question_en": "What are railroad cars professionally braked with?",
"options_str_en": "A: Strappy sandal, B: Leather boot, C: Shower sandals, D: Brake shoe"
}
```
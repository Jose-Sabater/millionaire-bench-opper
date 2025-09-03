import os
import asyncio
from opperai import Opper
from pydantic import BaseModel


opper = Opper(http_bearer=os.getenv("OPPER_API_KEY"))


class TranslationOutput(BaseModel):
    translated_text: str

async def translate(text):
    response = await opper.call_async(
        name="translate-ger-en",
        instructions="Translate the text from German to English, keep the formatting as close as possible. If there is not a perfect match, just translate the best you can (accounting for idioms, jokes, etc.). Keep same option order as in the original text.",
        output_schema=TranslationOutput,
        input=text,
        model="openai/gpt-5",
    )
    return response.json_payload["translated_text"]

# translate
import json

async def bounded_translate(text, semaphore):
    async with semaphore:
        return await translate(text)

async def translate_question_object(question_obj, semaphore):
    question_en, options_str_en = await asyncio.gather(
        bounded_translate(question_obj["question"], semaphore),
        bounded_translate(question_obj["options_str"], semaphore),
    )
    new_question = dict(question_obj)
    new_question["question_en"] = question_en
    new_question["options_str_en"] = options_str_en
    return new_question

async def build_translated_dataset(input_data, concurrency_limit=10):
    semaphore = asyncio.Semaphore(concurrency_limit)
    new_dataset = []
    for program in input_data:
        new_program = dict(program)
        questions = program.get("questions", [])
        if isinstance(questions, list):
            tasks = [asyncio.create_task(translate_question_object(q, semaphore)) for q in questions]
            new_program["questions"] = await asyncio.gather(*tasks)
        elif isinstance(questions, dict):
            tasks_map = {k: asyncio.create_task(translate_question_object(v, semaphore)) for k, v in questions.items()}
            new_program["questions"] = {k: await t for k, t in tasks_map.items()}
        else:
            new_program["questions"] = questions
        new_dataset.append(new_program)
    return new_dataset

async def main():
    with open("fragen_antworten_better.json", "r") as f:
        input_data = json.load(f)
        # input_data = input_data[:1]
    translated = await build_translated_dataset(input_data, concurrency_limit=25)
    with open("fragen_antworten_better_en.json", "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
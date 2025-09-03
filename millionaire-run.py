import os
import asyncio
from opperai import Opper
from typing import Literal, Dict, List, Any
from pydantic import BaseModel
import json
import datetime
from pathlib import Path

opper = Opper(http_bearer=os.getenv("OPPER_API_KEY"))

# Select models available: 
models = [
    # "anthropic/claude-3.5-haiku",
    # "anthropic/claude-sonnet-4",
    # "berget/gpt-oss-120b",
    "gcp/gemini-2.5-pro",
    # "gcp/gemini-2.5-flash",
    # "gcp/gemini-2.5-flash-lite",
    # "groq/gemma2-9b-it",
    # "groq/gpt-oss-20b",
    # "groq/llama-3.1-8b-instant",
    # "groq/moonshotai/kimi-k2-instruct",
    # "mistral/mistral-medium-2508-eu",
    # "openai/gpt-5",
    # "openai/gpt-5-mini",
    # "openai/gpt-5-nano",
    # "xai/grok-4",
]

# Load the millionaire questions data
with open("fragen_antworten_en.json", "r") as f:
    data = json.load(f)

# Who Wants to Be a Millionaire earnings ladder (German format)
EARNINGS_LADDER = {
    1: 50,  # Level 1: 50€
    2: 100,  # Level 2: 100
    3: 200,  # Level 3: 200€
    4: 300,  # Level 4: 300€
    5: 500,  # Level 5: 500€ (safe haven)
    6: 1000,  # Level 6: 1.000€
    7: 2000,  # Level 7: 2.000€
    8: 4000,  # Level 8: 4.000€
    9: 8000,  # Level 9: 8.000€
    10: 16000,  # Level 10: 16.000€ (safe haven)
    11: 32000,  # Level 11: 32.000€
    12: 64000,  # Level 12: 64.000€
    13: 125000,  # Level 13: 125.000€
    14: 500000,  # Level 14: 500.000€
    15: 1000000,  # Level 15: 1.000.000€
}

# No safe havens - earnings are simply based on questions answered correctly


class ModelResponse(BaseModel):
    response_letter: Literal["A", "B", "C", "D"]


class QuestionResult(BaseModel):
    question_level: int
    question_text: str
    question_text_en: str
    options: Dict[str, str]
    correct_answer: str
    generated_answer: str
    is_correct: bool
    earnings_at_level: int


class ProgramResult(BaseModel):
    program_id: int
    questions_answered: int
    final_earnings: int
    eliminated_at_level: int | None
    question_results: List[QuestionResult]


class ModelEvaluation(BaseModel):
    model: str
    total_programs: int
    successful_programs: int
    total_earnings: int
    average_earnings: float
    programs: List[ProgramResult]
    timestamp: str


# Semaphore for concurrency control
semaphore = asyncio.Semaphore(15)


async def generate_response(
    question_data: Dict[str, Any], model: str, parent_span_id: str
) -> tuple[str, str]:
    """Generate response for a single question using the specified model with retries."""
    # Format question for the model
    question_text = (
        f"{question_data['question']}\n\nOptions:\n{question_data['options_str']}"
    )

    max_retries = 3
    last_exception = None

    for attempt in range(
        max_retries + 1
    ):  # 0, 1, 2, 3 (total of 4 attempts including first)
        try:
            response = await opper.call_async(
                name="millionaire-question",
                instructions="You are playing 'Who Wants to Be a Millionaire' in german. Read the question carefully and select the correct answer letter (A, B, C, or D). Only respond with the letter.",
                output_schema=ModelResponse,
                input=question_text,
                model=model,
                parent_span_id=parent_span_id,
            )
            return response.json_payload["response_letter"], response.span_id

        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                print(
                    f"Attempt {attempt + 1} failed for {model}, retrying... Error: {e}"
                )
                # Small delay before retry
                await asyncio.sleep(1)
            else:
                print(
                    f"All {max_retries + 1} attempts failed for {model}. Final error: {e}"
                )
                return "ERROR", None  # Return ERROR and no span_id


def calculate_final_earnings(questions_answered: int, eliminated: bool) -> int:
    """Calculate final earnings based on questions answered correctly."""
    if questions_answered == 0:
        return 0

    # Simply return earnings for the last question answered correctly
    return EARNINGS_LADDER[questions_answered]


async def run_single_program(
    program_data: Dict[str, Any], model: str, parent_span_id: str
) -> ProgramResult:
    """Run a single millionaire program for a model."""
    async with semaphore:
        program_id = program_data["program"]
        questions = program_data["questions"]
        question_results = []

        # Create a child span for this specific program
        program_span = await opper.spans.create_async(
            name=f"millionaire-program-{program_id}",
            input=f"Program {program_id} with {len(questions)} questions",
            parent_id=parent_span_id,
        )

        try:
            for i, question in enumerate(questions, 1):
                try:
                    generated_answer, question_span_id = await generate_response(
                        question, model, program_span.id
                    )
                    correct_answer = question["answer"]
                    # Treat ERROR as incorrect answer
                    is_correct = (
                        generated_answer == correct_answer
                        and generated_answer != "ERROR"
                    )

                    # Add metrics to the question span if we have a span_id
                    if question_span_id:
                        try:
                            # Add correctness metric (1 for correct, 0 for incorrect)
                            await opper.span_metrics.create_metric_async(
                                span_id=question_span_id,
                                dimension="question_correct",
                                value=1 if is_correct else 0,
                                comment=f"Question {i} correctness (1=correct, 0=incorrect)",
                            )

                            # Add earnings metric
                            earnings_at_level = EARNINGS_LADDER[i]
                            await opper.span_metrics.create_metric_async(
                                span_id=question_span_id,
                                dimension="potential_earnings",
                                value=earnings_at_level,
                                comment=f"Potential earnings at level {i}: eur {earnings_at_level:,}",
                            )

                            # Add question level metric
                            await opper.span_metrics.create_metric_async(
                                span_id=question_span_id,
                                dimension="question_level",
                                value=i,
                                comment=f"Question difficulty level (1-15)",
                            )

                        except Exception as metric_error:
                            print(
                                f"Failed to add metrics for question {i}: {metric_error}"
                            )

                    question_result = QuestionResult(
                        question_level=i,
                        question_text=question["question"],
                        question_text_en=question["question_en"],
                        options=question["options"],
                        correct_answer=correct_answer,
                        generated_answer=generated_answer,
                        is_correct=is_correct,
                        earnings_at_level=EARNINGS_LADDER[i],
                    )
                    question_results.append(question_result)

                    # If wrong answer, eliminate
                    if not is_correct:
                        final_earnings = calculate_final_earnings(
                            i - 1, eliminated=True
                        )
                        result = ProgramResult(
                            program_id=program_id,
                            questions_answered=i - 1,
                            final_earnings=final_earnings,
                            eliminated_at_level=i,
                            question_results=question_results,
                        )

                        # Update program span with elimination result
                        await opper.spans.update_async(
                            span_id=program_span.id,
                            output=f"Eliminated at question {i}. Final earnings: eur {final_earnings:,}",
                        )

                        # Add program-level metrics
                        try:
                            await opper.span_metrics.create_metric_async(
                                span_id=program_span.id,
                                dimension="final_earnings",
                                value=final_earnings,
                                comment=f"Final earnings for program {program_id}: eur {final_earnings:,}",
                            )
                            await opper.span_metrics.create_metric_async(
                                span_id=program_span.id,
                                dimension="questions_answered",
                                value=i - 1,
                                comment=f"Questions answered correctly before elimination",
                            )
                            await opper.span_metrics.create_metric_async(
                                span_id=program_span.id,
                                dimension="program_success",
                                value=0,
                                comment="Program eliminated (0=eliminated, 1=completed)",
                            )
                        except Exception as metric_error:
                            print(f"Failed to add program metrics: {metric_error}")

                        return result

                except Exception as e:
                    print(f"Error in program {program_id}, question {i}: {e}")
                    # Treat as elimination
                    final_earnings = calculate_final_earnings(i - 1, eliminated=True)
                    result = ProgramResult(
                        program_id=program_id,
                        questions_answered=i - 1,
                        final_earnings=final_earnings,
                        eliminated_at_level=i,
                        question_results=question_results,
                    )

                    # Update program span with error result
                    await opper.spans.update_async(
                        span_id=program_span.id,
                        output=f"Error at question {i}: {e}. Final earnings: eur {final_earnings:,}",
                    )

                    # Add program-level metrics for error case
                    try:
                        await opper.span_metrics.create_metric_async(
                            span_id=program_span.id,
                            dimension="final_earnings",
                            value=final_earnings,
                            comment=f"Final earnings for program {program_id}: eur {final_earnings:,}",
                        )
                        await opper.span_metrics.create_metric_async(
                            span_id=program_span.id,
                            dimension="questions_answered",
                            value=i - 1,
                            comment=f"Questions answered correctly before error",
                        )
                        await opper.span_metrics.create_metric_async(
                            span_id=program_span.id,
                            dimension="program_success",
                            value=0,
                            comment="Program failed due to error (0=eliminated, 1=completed)",
                        )
                    except Exception as metric_error:
                        print(f"Failed to add program metrics: {metric_error}")

                    return result

            # Completed all questions successfully
            final_earnings = calculate_final_earnings(len(questions), eliminated=False)
            result = ProgramResult(
                program_id=program_id,
                questions_answered=len(questions),
                final_earnings=final_earnings,
                eliminated_at_level=None,
                question_results=question_results,
            )

            # Update program span with success result
            await opper.spans.update_async(
                span_id=program_span.id,
                output=f"Completed all {len(questions)} questions! Final earnings: eur {final_earnings:,}",
            )

            # Add program-level metrics for successful completion
            try:
                await opper.span_metrics.create_metric_async(
                    span_id=program_span.id,
                    dimension="final_earnings",
                    value=final_earnings,
                    comment=f"Final earnings for program {program_id}: eur {final_earnings:,}",
                )
                await opper.span_metrics.create_metric_async(
                    span_id=program_span.id,
                    dimension="questions_answered",
                    value=len(questions),
                    comment=f"All {len(questions)} questions answered correctly",
                )
                await opper.span_metrics.create_metric_async(
                    span_id=program_span.id,
                    dimension="program_success",
                    value=1,
                    comment="Program completed successfully (0=eliminated, 1=completed)",
                )
            except Exception as metric_error:
                print(f"Failed to add program metrics: {metric_error}")

            return result

        except Exception as e:
            # Update program span with general error
            await opper.spans.update_async(
                span_id=program_span.id, output=f"Program failed with error: {e}"
            )
            raise


async def evaluate_model(model: str) -> ModelEvaluation:
    """Evaluate a single model across all 45 programs."""
    print(f"Starting evaluation for model: {model}")

    # Create parent span for this model evaluation
    parent_span = await opper.spans.create_async(
        name=f"millionaire-evaluation-{model.replace('/', '-')}",
        input=f"Evaluating {model} on {len(data)} millionaire programs",
    )

    # Run all programs concurrently
    tasks = []
    for program_data in data:
        task = run_single_program(program_data, model, parent_span.id)
        tasks.append(task)

    # Execute all programs with concurrency control
    program_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and count successes
    valid_results = []
    for result in program_results:
        if isinstance(result, Exception):
            print(f"Error in program: {result}")
        else:
            valid_results.append(result)

    # Calculate statistics
    total_programs = len(valid_results)
    successful_programs = len(
        [r for r in valid_results if r.eliminated_at_level is None]
    )
    total_earnings = sum(r.final_earnings for r in valid_results)
    average_earnings = total_earnings / total_programs if total_programs > 0 else 0

    # Update parent span with results
    await opper.spans.update_async(
        span_id=parent_span.id,
        output=f"Completed {total_programs} programs. Total earnings: eur {total_earnings:,}. Average: eur {average_earnings:.2f}",
    )

    evaluation = ModelEvaluation(
        model=model,
        total_programs=total_programs,
        successful_programs=successful_programs,
        total_earnings=total_earnings,
        average_earnings=average_earnings,
        programs=valid_results,
        timestamp=datetime.datetime.now().isoformat(),
    )

    print(
        f"Completed evaluation for {model}: {successful_programs}/{total_programs} programs successful, eur {total_earnings:,} total earnings"
    )
    return evaluation


def save_results(evaluation: ModelEvaluation):
    """Save evaluation results to JSON file."""
    results_dir = Path("millionaire_results")
    results_dir.mkdir(exist_ok=True)

    # Clean model name for filename
    safe_model_name = evaluation.model.replace("/", "_").replace(":", "_")
    filename = f"result_{safe_model_name}.json"
    filepath = results_dir / filename

    # Convert to dict for JSON serialization
    result_dict = evaluation.model_dump()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {filepath}")


async def run_all_evaluations():
    """Run evaluations for all models."""
    print(
        f"Starting millionaire benchmark evaluation with {len(models)} models and {len(data)} programs"
    )
    print(f"Concurrency: 15 simultaneous programs per model")
    print(f"Total questions to be asked: {len(models) * len(data) * 15}")

    all_results = []

    for model in models:
        try:
            evaluation = await evaluate_model(model)
            save_results(evaluation)
            all_results.append(evaluation)

            # Brief summary
            print(f"\n--- Summary for {model} ---")
            print(
                f"Programs completed successfully: {evaluation.successful_programs}/{evaluation.total_programs}"
            )
            print(f"Total earnings: eur {evaluation.total_earnings:,}")
            print(f"Average earnings per program: eur {evaluation.average_earnings:.2f}")
            print(
                f"Best program earnings: eur {max(r.final_earnings for r in evaluation.programs) if evaluation.programs else 0:,}"
            )

        except Exception as e:
            print(f"Failed to evaluate model {model}: {e}")
            continue

    # Generate summary report
    if all_results:
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")

        for evaluation in sorted(
            all_results, key=lambda x: x.average_earnings, reverse=True
        ):
            print(
                f"{evaluation.model:30} | Avg: eur {evaluation.average_earnings:8.2f} | Total: eur {evaluation.total_earnings:10,} | Success: {evaluation.successful_programs:2d}/{evaluation.total_programs}"
            )

        # Save combined results
        combined_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_models": len(all_results),
            "total_programs_per_model": len(data),
            "evaluations": [eval.model_dump() for eval in all_results],
        }

        with open(
            "millionaire_results/combined_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)

        # Create summarized leaderboard
        leaderboard = []
        for evaluation in sorted(
            all_results, key=lambda x: x.average_earnings, reverse=True
        ):
            leaderboard_entry = {
                "model": evaluation.model,
                "total_earnings": evaluation.total_earnings,
                "average_earnings": round(evaluation.average_earnings, 2),
                "successful_programs": evaluation.successful_programs,
                "total_programs": evaluation.total_programs,
                "success_rate": (
                    round(
                        (evaluation.successful_programs / evaluation.total_programs)
                        * 100,
                        1,
                    )
                    if evaluation.total_programs > 0
                    else 0.0
                ),
                "max_earnings_single_program": (
                    max(r.final_earnings for r in evaluation.programs)
                    if evaluation.programs
                    else 0
                ),
            }
            leaderboard.append(leaderboard_entry)

        leaderboard_summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_models_evaluated": len(all_results),
            "total_programs_per_model": len(data),
            "leaderboard": leaderboard,
        }

        with open("millionaire_results/leaderboard.json", "w", encoding="utf-8") as f:
            json.dump(leaderboard_summary, f, indent=2, ensure_ascii=False)

        print(f"\nCombined results saved to: millionaire_results/combined_results.json")
        print(f"Leaderboard saved to: millionaire_results/leaderboard.json")


if __name__ == "__main__":
    asyncio.run(run_all_evaluations())

#!/usr/bin/env python3
import json
import requests
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration - You can modify these values
SYSTEM_PROMPT = "You are a contestant on 'Who Wants to Be a Millionaire' and must answer questions in German. Think carefully and choose the best answer from the four options. Respond EXCLUSIVELY with a single letter: A, B, C, or D. No other explanation, just the letter! Example: If A is the correct answer, respond only: A"
MODEL_NAME = "llama-3.2-3b-instruct"
LLM_SERVER_URL = "http://localhost:1234/v1/chat/completions"
LLM_API_KEY = "None"  # Optional, provide API key if required by your server
TEMPERATURE = 0.6
TOP_K = 40
TOP_P = 0.9
MIN_P = 0
TOTAL_PARAMETERS = "3B"  # e.g., "7B", "70B", etc.
ACTIVE_PARAMETERS = "3B"  # e.g., "7B", "35B", etc. (for MoE models)
CONCURRENCY_LEVEL = 1  # Number of concurrent game rounds (default of 1 for sequential play, set 45 to start all games at once)

# Prize amounts for each level
PRIZE_AMOUNTS = {
    1: "50€",
    2: "100€",
    3: "200€",
    4: "300€",
    5: "500€",
    6: "1.000€",
    7: "2.000€",
    8: "4.000€",
    9: "8.000€",
    10: "16.000€",
    11: "32.000€",
    12: "64.000€",
    13: "125.000€",
    14: "500.000€",
    15: "1.000.000€"
}

def load_questions(filename):
    """Load questions from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_user_choice():
    """Ask user which question number to start with"""
    try:
        choice = input("Which question should be asked first? (Default: 1, 0 for all 45 questions): ").strip()
        if choice == "":
            return 1
        return int(choice)
    except ValueError:
        print("Invalid input. Using question #1.")
        return 1

def calculate_average_amount(rounds):
    """Calculate average final amount from rounds"""
    if not rounds:
        return "0€"
    
    # Convert amounts to numeric values for calculation
    amount_mapping = {
        "0€": 0,
        "50€": 50,
        "100€": 100,
        "200€": 200,
        "300€": 300,
        "500€": 500,
        "1.000€": 1000,
        "2.000€": 2000,
        "4.000€": 4000,
        "8.000€": 8000,
        "16.000€": 16000,
        "32.000€": 32000,
        "64.000€": 64000,
        "125.000€": 125000,
        "500.000€": 500000,
        "1.000.000€": 1000000
    }
    
    total = sum(amount_mapping[round_data["final_amount"]] for round_data in rounds)
    average = total / len(rounds)
    
    # Convert back to formatted string with full numbers
    if average >= 1000000:
        return f"{average:,.1f}€".replace(",", ".")
    elif average >= 1000:
        return f"{average:,.0f}€".replace(",", ".")
    else:
        return f"{int(average)}€"

def play_single_game(questions, start_question=1, silent=False):
    """Play a single game with one question"""
    current_level = 1
    question_number = start_question
    correct_answers = 0
    
    if not silent:
        print(f"Starting the game with question #{question_number}")
        print("=" * 50)
    
    while current_level <= 15:
        # Get question for current level
        if str(current_level) not in questions:
            print(f"Error: No questions for level {current_level}")
            break
            
        level_questions = questions[str(current_level)]
        
        if question_number > len(level_questions):
            print(f"Error: Question #{question_number} does not exist in level {current_level}")
            break
            
        question_data = level_questions[question_number-1]
        question_text = question_data[0]
        options = question_data[1:5]
        correct_answer = question_data[5]
        
        if not silent:
            print(f"\nLevel {current_level} - {PRIZE_AMOUNTS[current_level]}")
            print(f"Question #{question_number}: {question_text}")
            print(f"A: {options[0]}")
            print(f"B: {options[1]}")
            print(f"C: {options[2]}")
            print(f"D: {options[3]}")
        
        # Format prompt for LLM
        prompt = format_question(question_data)
        
        # Get LLM response
        if not silent:
            print("\nWaiting for AI model response...")
        start_time = time.time()
        llm_answer = get_llm_response(prompt, SYSTEM_PROMPT, MODEL_NAME)
        response_time = time.time() - start_time
        
        if not silent:
            print(f"AI model response: {llm_answer} (in {response_time:.2f} seconds)")
        
        # Convert correct answer text to letter
        correct_options = options
        try:
            correct_index = correct_options.index(correct_answer)
            correct_letter = ['A', 'B', 'C', 'D'][correct_index]
        except ValueError:
            print(f"Error: Correct answer '{correct_answer}' not found in options")
            break
        
        if not silent:
            print(f"Correct answer: {correct_letter} ({correct_answer})")
        
        # Check if answer is correct
        if llm_answer == correct_letter:
            if not silent:
                print("✓ Correct!")
            correct_answers += 1
            current_level += 1
        else:
            if not silent:
                print(f"✗ Wrong! (AI said {llm_answer}, correct was {correct_letter})")
            break
    
    # Game result
    if not silent:
        print("\n" + "=" * 50)
        print("GAME OVER")
        print("=" * 50)
    
    return {
        "start_question": start_question,
        "correct_answers": correct_answers,
        "final_amount": PRIZE_AMOUNTS[correct_answers] if correct_answers > 0 else "0€"
    }

def play_all_games(questions):
    """Play all 45 questions with configurable concurrency"""
    results = []
    
    if CONCURRENCY_LEVEL > 1:
        print(f"Starting all 45 rounds with concurrency level: {CONCURRENCY_LEVEL}")
    else:
        print("Starting all 45 questions sequentially...")
    print("=" * 50)
    
    if CONCURRENCY_LEVEL == 1:
        # Sequential execution (original behavior)
        for question_num in range(1, 46):
            print(f"\nQUESTION {question_num}/45")
            print("-" * 30)
            
            # Play single game with this question
            result = play_single_game(questions, question_num)
            result["question_number"] = question_num
            results.append(result)
            
            # Clear context by adding a visual separator
            print("\n" + "=" * 50)
            print("CONTEXT CLEARED - NEW GAME")
            print("=" * 50)
    else:
        # Concurrent execution using thread pool
        def run_game(q_num):
            res = play_single_game(questions, q_num, silent=True)  # Silent mode for concurrent
            res["question_number"] = q_num
            return res
        
        with ThreadPoolExecutor(max_workers=CONCURRENCY_LEVEL) as executor:
            # Submit all tasks
            futures = {executor.submit(run_game, num): num for num in range(1, 46)}
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    print(f"[Round {result['question_number']:2d}/45] Completed - Level {result['correct_answers']:2d} - Won {result['final_amount']:>10s} ({completed_count}/45 done)")
                except Exception as e:
                    q_num = futures[future]
                    completed_count += 1
                    print(f"[Round {q_num:2d}/45] Error: {e} ({completed_count}/45 done)")
                    results.append({
                        "question_number": q_num,
                        "start_question": q_num,
                        "correct_answers": 0,
                        "final_amount": "0€"
                    })
        
        # Sort results to maintain order
        results.sort(key=lambda x: x["question_number"])
    
    return results

def format_question(question_data):
    """Format question and options for LLM"""
    question = question_data[0]
    options = question_data[1:5]
    return f"{question}\nA: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}"

def get_llm_response(prompt, system_prompt, model_name):
    """Send prompt to LLM and get response"""
    try:
        # Prepare headers with API key if provided
        headers = {"Content-Type": "application/json"}
        if LLM_API_KEY and LLM_API_KEY != "None" and LLM_API_KEY != "":
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "min_p": MIN_P
        }
        
        response = requests.post(LLM_SERVER_URL, json=data, headers=headers, timeout=60)
        if response.status_code == 200:
            # Extract content and handle reasoning models that include thinking before the answer
            content = response.json()["choices"][0]["message"]["content"]
            # For reasoning models, extract the part after the last thinking separator
            result = content.split('</scratchpad>')[-1].strip(' \t\n\r')
            # Extract just the letter if there's more text
            # First check if the result is directly a letter
            if result.upper() in ['A', 'B', 'C', 'D']:
                return result.upper()
            
            # If not, try to find a letter in the response
            for char in result:
                if char.upper() in ['A', 'B', 'C', 'D']:
                    return char.upper()
            
            # If we still haven't found a valid answer, return error
            print(f"Warning: Unexpected response format received: '{result}'")
            return "INVALID"
        else:
            print(f"Error: Server responded with status {response.status_code}")
            return "ERROR"
    except Exception as e:
        print(f"Error in request: {e}")
        return "ERROR"

def load_model_results(model_name):
    """Load existing results for a model"""
    # Create a safe filename from the model name
    safe_model_name = model_name.replace("/", "-")
    result_filename = f"result_{safe_model_name}.json"
    
    try:
        with open(result_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Add model parameters if they don't exist
            if "model_parameters" not in data:
                data["model_parameters"] = {
                    "temperature": TEMPERATURE,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "min_p": MIN_P
                }
            # Update model parameters to current values
            else:
                data["model_parameters"] = {
                    "temperature": TEMPERATURE,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "min_p": MIN_P
                }
            # Add metadata fields if they don't exist
            if "total_parameters" not in data:
                data["total_parameters"] = TOTAL_PARAMETERS
            if "active_parameters" not in data:
                data["active_parameters"] = ACTIVE_PARAMETERS
            return data, result_filename
    except FileNotFoundError:
        # Create new results file
        initial_data = {
            "model": model_name,
            "model_parameters": {
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "min_p": MIN_P
            },
            "total_parameters": TOTAL_PARAMETERS,
            "active_parameters": ACTIVE_PARAMETERS,
            "rounds": []
        }
        return initial_data, result_filename

def save_model_results(results_data, filename):
    """Save model results to file"""
    # Sort rounds by start_question before saving
    if "rounds" in results_data:
        results_data["rounds"].sort(key=lambda x: x["start_question"])
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

def calculate_average_correctness_percentage(rounds):
    """Calculate average correctness percentage from rounds"""
    if not rounds:
        return 0.0
    
    total_questions = sum(round_data["correct_answers"] + 1 if round_data["correct_answers"] < 15 else 15 for round_data in rounds)
    total_correct = sum(round_data["correct_answers"] for round_data in rounds)
    
    if total_questions == 0:
        return 0.0
        
    return round((total_correct / total_questions) * 100, 2)

def play_game(questions, start_question=1):
    """Main game loop - handles both single question and all 45 questions"""
    if start_question == 0:
        # Play all 45 questions
        game_results = play_all_games(questions)
    else:
        # Play single game
        single_result = play_single_game(questions, start_question)
        game_results = [single_result]
    
    # Save results to model-specific file
    results_data, result_filename = load_model_results(MODEL_NAME)
    
    # Handle deduplication - remove existing rounds with same start_question
    if start_question == 0:
        # For all-games mode, we don't deduplicate as each question is unique
        pass
    else:
        # For single game mode, remove existing rounds with same start_question (but keep all-games rounds)
        results_data["rounds"] = [
            r for r in results_data["rounds"] 
            if r.get("start_question") != start_question or r.get("question_number") is not None
        ]
    
    # Add new rounds
    for i, result in enumerate(game_results):
        round_data = {
            "start_question": result["start_question"],
            "correct_answers": result["correct_answers"],
            "final_amount": result["final_amount"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add question_number for all-games mode
        if start_question == 0:
            round_data["question_number"] = result["question_number"]
            
        results_data["rounds"].append(round_data)
    
    # Sort rounds by start_question
    results_data["rounds"].sort(key=lambda x: x["start_question"])
    
    # Calculate and add average final amount and correctness percentage
    results_data["average_final_amount"] = calculate_average_amount(results_data["rounds"])
    results_data["average_correctness_percentage"] = calculate_average_correctness_percentage(results_data["rounds"])
    
    # Count million wins (questions with 15 correct answers)
    million_wins = sum(1 for r in results_data["rounds"] if r["correct_answers"] == 15)
    
    # Save updated results
    save_model_results(results_data, result_filename)
    
    # Improved summary display
    print(f"Million Wins: {million_wins}")
    print(f"Results saved in: {result_filename}")
    print(f"Average winnings: {results_data['average_final_amount']}")
    print(f"Parameters: T:{TEMPERATURE}, K:{TOP_K}, P:{TOP_P}, Min:{MIN_P}")
    
    return game_results

def main():
    """Main function"""
    print("Who Wants to Be a Millionaire - LLM Benchmark")
    print("=" * 50)
    
    # Load questions
    try:
        questions = load_questions("fragen_antworten.json")
        print(f"Questions loaded: {len(questions)} levels")
    except Exception as e:
        print(f"Error loading questions: {e}")
        return
    
    # Get user choice
    start_question = get_user_choice()
    
    # Play game
    result = play_game(questions, start_question)
    
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()

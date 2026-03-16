"""
Data Preparation Script for G-MSRA.
Downloads and formats benchmark datasets into the unified JSON format
expected by training and evaluation scripts.

Supported datasets:
- LoCoMo: Long-term conversational memory benchmark
- LongMemEval: Long-term memory evaluation
- ALFWorld: Agent task environment
- Evo-Memory: Memory evolution benchmark

Usage:
    python scripts/prepare_data.py --output_dir data
    python scripts/prepare_data.py --dataset locomo --output_dir data
"""

import argparse
import json
import os
import sys

from loguru import logger


def prepare_locomo(output_dir: str):
    """Download and format LoCoMo dataset.

    LoCoMo is a long-term conversational memory benchmark with
    multi-session dialogues and QA evaluation.

    Source: https://github.com/LLM-Evaluation/LoCoMo
    """
    output_train = os.path.join(output_dir, "locomo_train.json")
    output_test = os.path.join(output_dir, "locomo_test.json")

    if os.path.exists(output_train) and os.path.exists(output_test):
        logger.info("LoCoMo data already exists, skipping download")
        return

    try:
        # Try loading from HuggingFace datasets
        from datasets import load_dataset
        logger.info("Downloading LoCoMo from HuggingFace...")
        ds = load_dataset("locomo-bench/locomo", trust_remote_code=True)

        train_data = _format_locomo_split(ds.get("train", ds.get("test", [])))
        test_data = _format_locomo_split(ds.get("test", ds.get("validation", [])))

        # If only one split, do 80/20 split
        if not test_data and train_data:
            split_idx = int(len(train_data) * 0.8)
            test_data = train_data[split_idx:]
            train_data = train_data[:split_idx]

    except Exception as e:
        logger.warning(f"Could not load LoCoMo from HuggingFace: {e}")
        logger.info("Generating synthetic LoCoMo-format data for development...")
        train_data, test_data = _generate_synthetic_locomo()

    # Save
    _save_json(train_data, output_train)
    _save_json(test_data, output_test)
    logger.info(f"LoCoMo: {len(train_data)} train, {len(test_data)} test episodes saved")


def _format_locomo_split(split) -> list[dict]:
    """Convert LoCoMo HuggingFace format to our unified format."""
    formatted = []
    for item in split:
        # Extract conversation events and QA pairs
        events = []
        if "conversations" in item:
            for conv in item["conversations"]:
                if isinstance(conv, dict):
                    role = conv.get("role", "user")
                    content = conv.get("content", "")
                    events.append(f"{role.capitalize()} says: {content}")
                elif isinstance(conv, str):
                    events.append(conv)
        elif "dialogue" in item:
            events = item["dialogue"] if isinstance(item["dialogue"], list) else [item["dialogue"]]
        elif "events" in item:
            events = item["events"]

        # Extract QA
        question = item.get("question", item.get("query", ""))
        answer = item.get("answer", item.get("response", ""))
        category = item.get("category", item.get("type", "unknown"))

        if events and question and answer:
            formatted.append({
                "events": events,
                "question": question,
                "answer": answer,
                "category": category,
            })
    return formatted


def _generate_synthetic_locomo() -> tuple[list[dict], list[dict]]:
    """Generate synthetic LoCoMo-style data for development/testing."""
    scenarios = [
        # Information extraction
        {
            "events": [
                "User says: Hi, I'm Zhang Wei, I work as a data scientist at Huawei.",
                "User says: I mainly use Python and PySpark for big data analysis.",
                "User says: I live in Shenzhen with my wife and our son.",
            ],
            "question": "What programming tools does the user use for work?",
            "answer": "Python and PySpark",
            "category": "information_extraction",
        },
        {
            "events": [
                "User says: I just adopted a golden retriever puppy named Coco.",
                "User says: Coco is 3 months old and loves to play fetch.",
                "User says: I also have an older cat named Whiskers.",
            ],
            "question": "What pets does the user have?",
            "answer": "A golden retriever puppy named Coco and a cat named Whiskers",
            "category": "information_extraction",
        },
        # Multi-session reasoning
        {
            "events": [
                "User says: I'm starting a new diet - going fully vegan.",
                "User says: Actually, my doctor said I should add eggs for protein.",
                "User says: So I guess I'm more of a lacto-ovo vegetarian now.",
            ],
            "question": "What is the user's current diet?",
            "answer": "Lacto-ovo vegetarian",
            "category": "multi_session_reasoning",
        },
        {
            "events": [
                "User says: I just got a job offer from Google.",
                "User says: But then Microsoft offered me a better package.",
                "User says: I decided to accept the Microsoft offer.",
            ],
            "question": "Which company did the user choose?",
            "answer": "Microsoft",
            "category": "multi_session_reasoning",
        },
        # Knowledge update
        {
            "events": [
                "User says: I live in Beijing, Chaoyang district.",
                "User says: I've been thinking about moving to Hangzhou.",
                "User says: I finally moved to Hangzhou last week!",
            ],
            "question": "Where does the user currently live?",
            "answer": "Hangzhou",
            "category": "knowledge_update",
        },
        {
            "events": [
                "User says: My phone number is 138-0000-1234.",
                "User says: I got a new SIM card.",
                "User says: My new number is 159-9999-5678.",
            ],
            "question": "What is the user's current phone number?",
            "answer": "159-9999-5678",
            "category": "knowledge_update",
        },
        # Temporal reasoning
        {
            "events": [
                "User says: I have a meeting with clients on Monday at 10am.",
                "User says: The meeting got moved to Wednesday at 2pm.",
                "User says: I need to prepare the quarterly report before the meeting.",
            ],
            "question": "When is the user's meeting with clients?",
            "answer": "Wednesday at 2pm",
            "category": "temporal_reasoning",
        },
        {
            "events": [
                "User says: My daughter's birthday is next Saturday.",
                "User says: She's turning 6 this year.",
                "User says: We're planning a party at the park.",
            ],
            "question": "How old will the user's daughter be?",
            "answer": "6",
            "category": "temporal_reasoning",
        },
        # Preference tracking
        {
            "events": [
                "User says: I prefer concise responses without too much detail.",
                "User says: When we discuss code, I want detailed explanations though.",
                "User says: Also, I like examples in Python, not Java.",
            ],
            "question": "How does the user want code discussions to be?",
            "answer": "Detailed explanations with Python examples",
            "category": "preference",
        },
        {
            "events": [
                "User says: I love morning coffee, usually a cappuccino.",
                "User says: I've switched to matcha lattes recently.",
                "User says: Actually I'm trying to cut caffeine entirely.",
            ],
            "question": "What is the user's current drink preference?",
            "answer": "The user is trying to cut caffeine entirely",
            "category": "preference",
        },
        # Abstain (unanswerable)
        {
            "events": [
                "User says: I work in the tech industry.",
                "User says: I enjoy going to concerts.",
                "User says: My favorite band is Radiohead.",
            ],
            "question": "What is the user's salary?",
            "answer": "Unknown - not mentioned in conversations",
            "category": "abstain",
        },
    ]

    # Extend with variations
    extended = scenarios * 5  # Repeat with shuffle
    import random
    random.seed(42)
    random.shuffle(extended)

    split_idx = int(len(extended) * 0.8)
    return extended[:split_idx], extended[split_idx:]


def prepare_longmemeval(output_dir: str):
    """Download and format LongMemEval dataset."""
    output_path = os.path.join(output_dir, "longmemeval_test.json")

    if os.path.exists(output_path):
        logger.info("LongMemEval data already exists, skipping")
        return

    try:
        from datasets import load_dataset
        logger.info("Downloading LongMemEval...")
        ds = load_dataset("LongMemEval/LongMemEval", trust_remote_code=True)

        test_data = []
        for item in ds.get("test", []):
            events = item.get("dialogue_history", item.get("events", []))
            if isinstance(events, str):
                events = events.split("\n")
            test_data.append({
                "events": events,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", item.get("type", "unknown")),
            })
    except Exception as e:
        logger.warning(f"Could not load LongMemEval: {e}")
        logger.info("Generating synthetic LongMemEval-format data...")
        _, test_data = _generate_synthetic_locomo()  # Reuse format
        for item in test_data:
            item["category"] = "synthetic"

    _save_json(test_data, output_path)
    logger.info(f"LongMemEval: {len(test_data)} test examples saved")


def prepare_alfworld(output_dir: str):
    """Prepare ALFWorld task data.

    ALFWorld is a text-based game environment for embodied agents.
    Source: https://github.com/alfworld/alfworld
    """
    output_path = os.path.join(output_dir, "alfworld_tasks.json")

    if os.path.exists(output_path):
        logger.info("ALFWorld data already exists, skipping")
        return

    # Generate synthetic ALFWorld-style tasks
    logger.info("Generating synthetic ALFWorld-format tasks for development...")

    task_types = [
        ("put", "Put the {obj} on the {recep}."),
        ("clean", "Clean the {obj} and put it on the {recep}."),
        ("heat", "Heat the {obj} and put it on the {recep}."),
        ("cool", "Cool the {obj} and put it on the {recep}."),
        ("examine", "Examine the {obj} using the {recep}."),
        ("pick_two", "Put two {obj}s on the {recep}."),
    ]

    objects = ["mug", "apple", "book", "pen", "plate", "knife", "fork",
               "bottle", "bowl", "cup", "cloth", "sponge", "candle"]
    receptacles = ["desk", "table", "shelf", "counter", "cabinet",
                   "drawer", "fridge", "microwave", "sink"]
    rooms = ["kitchen", "living room", "bedroom", "bathroom", "study"]

    import random
    random.seed(42)
    tasks = []

    for i in range(200):
        task_type, template = random.choice(task_types)
        obj = random.choice(objects)
        recep = random.choice(receptacles)
        room = random.choice(rooms)

        instruction = template.format(obj=obj, recep=recep)
        events = [
            f"You are in the {room}. You see a {recep} with some items on it.",
            f"You notice a {obj} on the floor near the {recep}.",
            f"There is also a {random.choice(objects)} on the {random.choice(receptacles)}.",
        ]

        tasks.append({
            "instruction": instruction,
            "events": events,
            "type": task_type,
            "context": f"Complete household task: {instruction}",
            "env_kwargs": {
                "task_result": {
                    "success": random.random() > 0.6,
                    "partial_score": random.uniform(0.2, 0.8),
                    "steps_taken": random.randint(3, 20),
                    "max_steps": 30,
                }
            },
        })

    _save_json(tasks, output_path)
    logger.info(f"ALFWorld: {len(tasks)} tasks saved")


def prepare_evomemory(output_dir: str):
    """Prepare Evo-Memory benchmark data."""
    output_path = os.path.join(output_dir, "evomemory_test.json")

    if os.path.exists(output_path):
        logger.info("Evo-Memory data already exists, skipping")
        return

    # Generate synthetic data based on Evo-Memory paper format
    logger.info("Generating synthetic Evo-Memory-format data...")

    import random
    random.seed(42)

    tasks = []
    for i in range(100):
        num_events = random.randint(5, 15)
        events = []
        facts = {}

        for j in range(num_events):
            fact_type = random.choice(["location", "job", "preference", "relation"])
            if fact_type == "location":
                city = random.choice(["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Chengdu"])
                events.append(f"User says: I{'m moving' if j > 0 else ' live'} in {city}.")
                facts["location"] = city
            elif fact_type == "job":
                company = random.choice(["Google", "Microsoft", "Alibaba", "ByteDance", "Huawei"])
                events.append(f"User says: I{'m switching to' if j > 0 else ' work at'} {company}.")
                facts["company"] = company
            elif fact_type == "preference":
                lang = random.choice(["Python", "Java", "Rust", "Go", "TypeScript"])
                events.append(f"User says: I{'ve switched to' if j > 0 else ' prefer'} {lang}.")
                facts["language"] = lang
            else:
                events.append(f"User says: My friend Alex visited last week.")

        # Question about latest state
        if "location" in facts:
            q, a = "Where does the user currently live?", facts["location"]
        elif "company" in facts:
            q, a = "Where does the user work?", facts["company"]
        else:
            q, a = "What programming language does the user prefer?", facts.get("language", "unknown")

        tasks.append({
            "events": events,
            "question": q,
            "answer": a,
            "category": "evolution_tracking",
            "num_updates": num_events,
        })

    _save_json(tasks, output_path)
    logger.info(f"Evo-Memory: {len(tasks)} test examples saved")


def _save_json(data, path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    datasets_to_prepare = {
        "locomo": prepare_locomo,
        "longmemeval": prepare_longmemeval,
        "alfworld": prepare_alfworld,
        "evomemory": prepare_evomemory,
    }

    if args.dataset:
        if args.dataset in datasets_to_prepare:
            datasets_to_prepare[args.dataset](args.output_dir)
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            sys.exit(1)
    else:
        logger.info("Preparing all datasets...")
        for name, prepare_fn in datasets_to_prepare.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Preparing: {name}")
            logger.info(f"{'='*50}")
            prepare_fn(args.output_dir)

    logger.info(f"\nAll datasets prepared in {args.output_dir}/")
    # List generated files
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".json"):
            size = os.path.getsize(os.path.join(args.output_dir, f))
            logger.info(f"  {f}: {size/1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Data Preparation")
    parser.add_argument("--output_dir", default="data",
                        help="Output directory for formatted datasets")
    parser.add_argument("--dataset", default=None,
                        choices=["locomo", "longmemeval", "alfworld", "evomemory"],
                        help="Specific dataset to prepare (default: all)")
    args = parser.parse_args()
    main(args)

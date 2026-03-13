"""
End-to-end test: verifiers pipeline with DAT-Bench environment.

Runs a small eval (3 examples, 1 rollout each) against a live model,
then validates the output structure and scores.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Ensure paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environments" / "dat_bench"))

# Set GloVe paths
os.environ.setdefault(
    "GLOVE_PATH",
    "/home/taf/Workspace/benchmarks/benchmarks/divergent_thinking/divergent-association-task/glove.840B.300d.txt",
)
os.environ.setdefault(
    "WORDS_PATH",
    "/home/taf/Workspace/benchmarks/benchmarks/divergent_thinking/divergent-association-task/words.txt",
)

# Load .env for API keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import verifiers as vf
from dat_bench import load_environment


async def main():
    print("=" * 60)
    print("DAT-Bench End-to-End Verifiers Pipeline Test")
    print("=" * 60)

    # 1. Load environment with fixture eval dataset
    print("\n[1] Loading environment...")
    env = load_environment(
        strategy="my_prompt",
        num_examples=3,
        use_fixture_eval=True,
    )
    print(f"    Train: {len(env.dataset)} rows")
    print(f"    Eval:  {len(env.eval_dataset)} rows (fixtures)")
    print(f"    Rubric funcs: {env.rubric._get_reward_func_names()}")

    # 2. Create client
    print("\n[2] Creating client...")
    model = "gpt-4.1-mini"
    config = vf.ClientConfig(
        api_key_var="OPENAI_API_KEY",
        api_base_url="https://api.openai.com/v1",
    )
    client = vf.OpenAIChatCompletionsClient(config)
    print(f"    Model: {model}")
    print(f"    Client: {type(client).__name__}")

    # 3. Build rollout inputs from eval dataset (just 3 to keep it fast)
    print("\n[3] Building rollout inputs...")
    n_examples = 3
    eval_ds = env.eval_dataset.select(range(n_examples))

    inputs = []
    for i in range(len(eval_ds)):
        row = eval_ds[i]
        inputs.append({
            "prompt": row["prompt"],
            "example_id": row["example_id"],
            "task": row["task"],
            "answer": row["answer"],
            "info": row["info"],
        })

    for inp in inputs:
        tier = inp["info"].get("fixture_tier", "?")
        print(f"    [{inp['example_id']}] tier={tier}")

    # 4. Run generation
    print(f"\n[4] Running {n_examples} rollouts...")
    result = await env.generate(
        inputs=inputs,
        client=client,
        model=model,
        sampling_args={"temperature": 0.7, "max_tokens": 500},
        max_concurrent=3,
    )

    outputs = result["outputs"]
    metadata = result["metadata"]

    print(f"    Completed: {len(outputs)} rollouts")
    print(f"    Avg reward: {metadata['avg_reward']:.3f}")
    print(f"    Avg error: {metadata['avg_error']:.3f}")

    # 5. Validate outputs
    print("\n[5] Validating outputs...")
    all_ok = True

    for out in outputs:
        eid = out["example_id"]
        reward = out["reward"]
        metrics = out["metrics"]
        info = out.get("info", {})
        tier = info.get("fixture_tier", "?")
        completion = out.get("completion", [])

        # Extract completion text
        comp_text = ""
        if completion:
            last = completion[-1]
            if isinstance(last, dict):
                comp_text = last.get("content", "")

        # Check structure
        assert "reward" in out, f"[{eid}] Missing reward"
        assert "metrics" in out, f"[{eid}] Missing metrics"
        assert "_dat_score" in metrics, f"[{eid}] Missing _dat_score metric"
        assert "_num_valid_words" in metrics, f"[{eid}] Missing _num_valid_words metric"
        assert "_raw_dat_score" in metrics, f"[{eid}] Missing _raw_dat_score metric"
        assert isinstance(reward, (int, float)), f"[{eid}] Reward not numeric: {type(reward)}"
        assert 0.0 <= reward <= 1.0, f"[{eid}] Reward out of range: {reward}"

        raw_score = metrics["_raw_dat_score"]
        valid_words = metrics["_num_valid_words"]
        dat_score = metrics["_dat_score"]

        status = "OK"
        if raw_score == 0.0 and valid_words < 7:
            status = "INVALID (too few words)"

        print(f"    [{eid}] tier={tier:<10} reward={reward:.3f}  raw={raw_score:.1f}  valid={int(valid_words)}  {status}")
        print(f"         completion: {comp_text[:80]}...")

    # 6. Check metadata structure
    print("\n[6] Metadata check...")
    assert "env_id" in metadata
    assert "model" in metadata
    assert "avg_reward" in metadata
    assert "avg_metrics" in metadata
    print(f"    env_id: {metadata['env_id']}")
    print(f"    model: {metadata['model']}")
    print(f"    avg_metrics: { {k: round(v, 3) for k, v in metadata['avg_metrics'].items()} }")

    print("\n" + "=" * 60)
    print("END-TO-END TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

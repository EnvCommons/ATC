"""Integration test for the ATC environment with trajectory logging.

Runs a single scenario using the OpenAI API and logs every tool call
to a .jsonl file for behavior inspection and debugging.
"""

import json
import asyncio
import os
import time

from openai import AsyncOpenAI
from openreward import OpenReward


async def main():
    or_client = OpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.4"
    ENV_NAME = "GeneralReasoning/atc"
    SPLIT = "train"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Connect to the environment
    environment = or_client.environments.get(name=ENV_NAME)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in {SPLIT} split")
    print(f"Tools available: {[t['function']['name'] for t in tools]}")

    # Open trajectory file
    trajectory_path = "trajectory.jsonl"
    trajectory_file = open(trajectory_path, "w")

    for task in tasks[:1]:  # Test first task only
        task_id = task.task_spec.get("id", "unknown")
        print(f"\n{'='*60}")
        print(f"Starting task: {task_id}")
        print(f"{'='*60}")

        # Create rollout for logging
        rollout = or_client.rollout.create(
            run_name="atc_test",
            rollout_name=f"test_{task_id}",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec,
        )

        async with environment.session(
            task=task, secrets={"openai_api_key": OPENAI_API_KEY}
        ) as session:
            # Get initial prompt
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            # Log initial prompt
            trajectory_file.write(json.dumps({
                "type": "prompt",
                "task_id": task_id,
                "content_length": len(prompt[0].text),
                "content_preview": prompt[0].text[:500],
                "timestamp": time.time(),
            }) + "\n")
            trajectory_file.flush()

            rollout.log_openai_response(
                message=input_list[0], is_finished=finished
            )

            turn = 0
            max_turns = 300  # Safety limit
            total_reward = 0.0

            while not finished and turn < max_turns:
                try:
                    # Call the model
                    response = await oai_client.responses.create(
                        model=MODEL_NAME,
                        tools=tools,
                        input=input_list,
                    )

                    rollout.log_openai_response(response.output[-1])
                    input_list += response.output

                    # Process tool calls
                    has_tool_call = False
                    for item in response.output:
                        if item.type == "function_call":
                            has_tool_call = True
                            args = json.loads(str(item.arguments))

                            # Execute tool
                            tool_result = await session.call_tool(
                                item.name, args
                            )

                            reward = tool_result.reward
                            finished = tool_result.finished
                            total_reward += reward if reward else 0

                            # Log to trajectory .jsonl
                            trajectory_entry = {
                                "type": "tool_call",
                                "turn": turn,
                                "tool": item.name,
                                "args": args,
                                "reward": reward,
                                "total_reward": total_reward,
                                "finished": finished,
                                "response_preview": tool_result.blocks[0].text[:400]
                                    if tool_result.blocks else "",
                                "timestamp": time.time(),
                            }
                            trajectory_file.write(
                                json.dumps(trajectory_entry) + "\n"
                            )
                            trajectory_file.flush()

                            # Add to input
                            input_list.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": tool_result.blocks[0].text
                                    if tool_result.blocks else "",
                            })

                            rollout.log_openai_response(
                                input_list[-1],
                                reward=reward,
                                is_finished=finished,
                            )

                            # Print progress
                            print(
                                f"  Turn {turn:3d} | "
                                f"Tool: {item.name:20s} | "
                                f"Reward: {reward:+.4f} | "
                                f"Total: {total_reward:+.4f}"
                            )

                            if finished:
                                print(f"\n  FINISHED! Final reward: {reward:.4f}")
                                break

                    if not has_tool_call:
                        # Model returned text without tool calls
                        text_content = ""
                        for item in response.output:
                            if hasattr(item, "text"):
                                text_content += item.text
                        trajectory_file.write(json.dumps({
                            "type": "model_text",
                            "turn": turn,
                            "text": text_content[:500],
                            "timestamp": time.time(),
                        }) + "\n")
                        trajectory_file.flush()

                except Exception as e:
                    print(f"  Error on turn {turn}: {e}")
                    trajectory_file.write(json.dumps({
                        "type": "error",
                        "turn": turn,
                        "error": str(e),
                        "timestamp": time.time(),
                    }) + "\n")
                    trajectory_file.flush()
                    break

                turn += 1

            # Final summary
            summary = {
                "type": "summary",
                "task_id": task_id,
                "total_turns": turn,
                "total_reward": total_reward,
                "finished": finished,
                "timestamp": time.time(),
            }
            trajectory_file.write(json.dumps(summary) + "\n")
            trajectory_file.flush()

            print(f"\n{'='*60}")
            print(f"Task {task_id} complete")
            print(f"  Turns: {turn}")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Finished: {finished}")
            print(f"  Trajectory: {trajectory_path}")
            print(f"{'='*60}")

    trajectory_file.close()
    print(f"\nTrajectory written to {trajectory_path}")


if __name__ == "__main__":
    asyncio.run(main())

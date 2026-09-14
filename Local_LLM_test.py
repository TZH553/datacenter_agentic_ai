from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_cluster_telemetry",
            "description": "Get the current data-centre telemetry.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[
        {
            "role": "user",
            "content": (
                "Check the current data-centre state. "
                "Use the get_cluster_telemetry tool."
            )
        }
    ],
    tools=tools,
    tool_choice="auto",
    temperature=0.1
)

print(response)

message = response.choices[0].message

print()
print("CONTENT:")
print(message.content)

print()
print("TOOL CALLS:")
print(message.tool_calls)
#!/bin/bash
for i in {20..30}; do
  words=$(for ((j=0; j<$i*10; j++)); do echo -n "apple "; done)
  cat << JSON > temp_q2.json
{
    "model": "deepseek-v4-flash",
    "messages": [
        {
            "role": "user",
            "content": "Repeat: $words"
        }
    ],
    "stream": false,
    "max_tokens": 10
}
JSON
  echo "Length $i * 10:"
  curl -s -X POST http://127.0.0.1:8001/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d @temp_q2.json | jq -r '.usage.prompt_tokens, .choices[0].message.content' | paste - -
done

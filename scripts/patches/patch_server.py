import re

with open('src/server_single.cpp', 'r') as f:
    content = f.read()

# 1. Update generate() signature
content = content.replace(
    'std::function<void(const std::string&)> on_token = nullptr,',
    'std::function<void(const std::string&, bool)> on_token = nullptr,'
)

# 2. Update on_token call in generate()
# Find the on_token call inside generate()
on_token_call = """                if (on_token) {
                    on_token(token_buffer);
                }"""
on_token_call_new = """                if (on_token) {
                    on_token(token_buffer, in_think_block);
                }"""
content = content.replace(on_token_call, on_token_call_new)

# 3. Handle think_block logic properly in generate()
# We need to NOT swallow the think tokens.
# Let's completely replace the think block handling inside the generate loop!
# DeepSeek-V4 uses "</think>" to end the reasoning block. The tokens themselves are generated one by one.
# So we can just use string matching for "</think>"!
think_logic_old = """            // Handle think block filtering
            if (think_start_id >= 0 && next_token == think_start_id) {
                in_think_block = true;
                // Forward the token but don't emit it
                forward_token(next_token, position);
                position++;
                continue;
            }
            if (think_end_id >= 0 && next_token == think_end_id) {
                in_think_block = false;
                think_block_ended = true;
                // Forward the token but don't emit it
                forward_token(next_token, position);
                position++;
                continue;
            }
            if (in_think_block) {
                // Inside think block \xe2\x80\x94 forward but don't emit
                forward_token(next_token, position);
                position++;
                continue;
            }"""

think_logic_new = """            // DeepSeek V4 might output "</think>" natively or we might need string matching.
            std::string token_text_raw = tokenizer_.decode({next_token});
            
            // Check if we hit the end of the think block token
            if (think_end_id >= 0 && next_token == think_end_id) {
                in_think_block = false;
                think_block_ended = true;
                forward_token(next_token, position);
                position++;
                continue;
            }

            // String matching for "</think>" in case it's not a single token
            if (in_think_block) {
                think_detect_buffer += token_text_raw;
                size_t pos = think_detect_buffer.find("</think>");
                if (pos != std::string::npos) {
                    in_think_block = false;
                    think_block_ended = true;
                    // We should strip "</think>" from the emitted text
                    std::string before = think_detect_buffer.substr(0, pos);
                    if (!before.empty() && on_token) {
                        on_token(before, true);
                    }
                    think_detect_buffer.clear();
                    forward_token(next_token, position);
                    position++;
                    continue;
                }
            }
"""

content = content.replace(think_logic_old, think_logic_new)

# We need to initialize in_think_block to true since we appended <think>\n to the prompt!
# And fix how token_text is used.
init_think_old = """        bool in_think_block = false;
        bool think_block_ended = false;
        std::string think_detect_buffer;"""

init_think_new = """        bool in_think_block = true; // We forcefully append <think>\\n to prompt
        bool think_block_ended = false;
        std::string think_detect_buffer;"""

content = content.replace(init_think_old, init_think_new)

# Replace the decoding section to handle in_think_block logic buffering
decoding_old = """            std::string token_text = tokenizer_.decode({next_token});

            // If we just exited a think block, skip leading newlines
            if (think_block_ended && !token_text.empty()) {
                size_t start = token_text.find_first_not_of("\\n\\r");
                if (start == std::string::npos) {
                    // All whitespace, skip
                    forward_token(next_token, position);
                    position++;
                    continue;
                }
                if (start > 0) token_text = token_text.substr(start);
                think_block_ended = false;
            }

            token_buffer += token_text;"""

decoding_new = """            std::string token_text = token_text_raw;

            // If we just exited a think block, skip leading newlines
            if (think_block_ended && !token_text.empty()) {
                size_t start = token_text.find_first_not_of("\\n\\r");
                if (start == std::string::npos) {
                    // All whitespace, skip
                    forward_token(next_token, position);
                    position++;
                    continue;
                }
                if (start > 0) token_text = token_text.substr(start);
                think_block_ended = false;
            }

            if (in_think_block) {
                // We emit reasoning tokens directly instead of using token_buffer to avoid complex utf8 splitting logic for now, 
                // but actually we should just use token_buffer.
                token_buffer += token_text;
            } else {
                token_buffer += token_text;
            }"""
content = content.replace(decoding_old, decoding_new)

# Update SSE endpoint callback
sse_old = """                        engine.generate(prompt, max_tokens, temperature, [&](const std::string& text) {
                            chunk["choices"][0]["delta"] = {{"content", text}};
                            std::string sse_chunk = "data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\\n\\n";
                            sink.write(sse_chunk.data(), sse_chunk.size());
                        }, repetition_penalty);"""

sse_new = """                        engine.generate(prompt, max_tokens, temperature, [&](const std::string& text, bool is_reasoning) {
                            if (is_reasoning) {
                                chunk["choices"][0]["delta"] = {{"reasoning_content", text}};
                            } else {
                                chunk["choices"][0]["delta"] = {{"content", text}};
                            }
                            std::string sse_chunk = "data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\\n\\n";
                            sink.write(sse_chunk.data(), sse_chunk.size());
                        }, repetition_penalty);"""

content = content.replace(sse_old, sse_new)

with open('src/server_single.cpp', 'w') as f:
    f.write(content)

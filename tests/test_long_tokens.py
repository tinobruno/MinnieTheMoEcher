from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V4-Flash-0731')
prompt = "The capital of Italy is Rome. It is not only the capital of the country but also one of the most historically and culturally significant cities in the world, known for its ancient landmarks like the Colosseum, the Roman Forum, and the Vatican City which is a separate entity within it. Rome has been the capital since 1871, after the unification of Italy as a kingdom with the unification of the various states and territories that were part of the country at that time under the rule of the King of Italy. Based on this text, what is the capital of Belgium?"
msgs = [{"role": "user", "content": prompt}]
formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
encoded = tok.encode(formatted)
print("Length:", len(encoded))

import urllib.request
import json
prompt = "Hello, this is a slightly longer sentence to see if it breaks."
# We can't directly call the tokenizer from python for moecher, unless we write a small cpp wrapper.

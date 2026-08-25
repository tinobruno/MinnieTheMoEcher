#include "src/server_single.cpp"
int main() {
    Engine engine("moecher_manifest.json");
    auto tokens = engine.tokenizer_.encode("What is 2+2?");
    for (int t : tokens) printf("%d ", t);
    printf("\n");
}

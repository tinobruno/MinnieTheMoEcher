#include <iostream>
#include <string>

using namespace std;

static const std::string THINK_END = "</think>";
static const std::string ASSISTANT = "<｜Assistant｜>";

int main() {
    string result = ASSISTANT;
    result += "<think>\n";
    cout << result << endl;
    return 0;
}

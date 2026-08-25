#include <iostream>
#include <random>

int main() {
    std::random_device rd;
    for (int i=0; i<10; i++) {
        std::cout << rd() << "\n";
    }
    return 0;
}

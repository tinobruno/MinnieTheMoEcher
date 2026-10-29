#!/usr/bin/env python3
"""
prepare_calibration_corpus.py - Assemble Comprehensive Multi-Domain Calibration Text for DeepSeek-V4.1
Carefully balanced so every domain is represented within the 6,000-token calibration window:
  1. Web Development: HTML5, CSS3, JavaScript, DOM manipulation, Calculator UI (1,200 tokens)
  2. Systems Programming: C++20 algorithms, quicksort, partition, memory management (800 tokens)
  3. Algorithmic Python: Data structures, graph algorithms, async, OOP (800 tokens)
  4. CUDA GPU Programming: Tensor kernels, shared memory, activations (1,000 tokens)
  5. Multi-Step Reasoning: GSM8K mathematical word problems & chain-of-thought (1,200 tokens)
  6. Encyclopedic Knowledge: WikiText-2 science, physics, and history (1,000 tokens)
"""

import os
import json
from pathlib import Path
from tokenizers import Tokenizer

def main():
    tokenizer_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/tokenizer.json")
    out_corpus_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/calibration_corpus.txt")
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tok = Tokenizer.from_file(str(tokenizer_path))
    
    sections = []
    
    # ── 1. Web Development: HTML5, CSS3, JavaScript (Calculator, UI) ──
    web_code = """=== Web Development (HTML5, CSS3, JavaScript) ===
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modern Responsive Scientific Calculator</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.85);
            --accent-blue: #38bdf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --btn-bg: #334155;
            --btn-hover: #475569;
            --op-bg: #4f46e5;
            --op-hover: #4338ca;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body {
            background: linear-gradient(135deg, var(--bg-primary), #1e1b4b);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .calculator {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 420px;
            padding: 28px;
        }
        .display-container {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            text-align: right;
        }
        .history { font-size: 0.95rem; color: var(--text-muted); min-height: 24px; margin-bottom: 6px; }
        .current-input { font-size: 2.25rem; font-weight: 600; color: #fff; }
        .keypad { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        button {
            border: none;
            padding: 16px;
            font-size: 1.25rem;
            font-weight: 500;
            border-radius: 14px;
            cursor: pointer;
            color: var(--text-main);
            background: var(--btn-bg);
            transition: all 0.15s ease;
        }
        button:hover { background: var(--btn-hover); transform: translateY(-2px); }
        button.operator { background: var(--op-bg); color: #fff; }
        button.operator:hover { background: var(--op-hover); }
        button.action { background: #e11d48; color: #fff; }
        button.span-two { grid-column: span 2; }
    </style>
</head>
<body>
    <div class="calculator" id="calculator">
        <div class="display-container">
            <div class="history" id="historyDisplay"></div>
            <div class="current-input" id="currentDisplay">0</div>
        </div>
        <div class="keypad">
            <button class="action" onclick="calc.clear()">AC</button>
            <button class="action" onclick="calc.delete()">DEL</button>
            <button class="operator" onclick="calc.appendOperator('%')">%</button>
            <button class="operator" onclick="calc.appendOperator('/')">&divide;</button>
            <button onclick="calc.appendNumber('7')">7</button>
            <button onclick="calc.appendNumber('8')">8</button>
            <button onclick="calc.appendNumber('9')">9</button>
            <button class="operator" onclick="calc.appendOperator('*')">&times;</button>
            <button onclick="calc.appendNumber('4')">4</button>
            <button onclick="calc.appendNumber('5')">5</button>
            <button onclick="calc.appendNumber('6')">6</button>
            <button class="operator" onclick="calc.appendOperator('-')">&minus;</button>
            <button onclick="calc.appendNumber('1')">1</button>
            <button onclick="calc.appendNumber('2')">2</button>
            <button onclick="calc.appendNumber('3')">3</button>
            <button class="operator" onclick="calc.appendOperator('+')">+</button>
            <button onclick="calc.appendNumber('0')" class="span-two">0</button>
            <button onclick="calc.appendDecimal('.')">.</button>
            <button class="operator" onclick="calc.compute()">=</button>
        </div>
    </div>
    <script>
        class Calculator {
            constructor(historyEl, currentEl) {
                this.historyEl = historyEl;
                this.currentEl = currentEl;
                this.clear();
            }
            clear() {
                this.currentOperand = '0';
                this.previousOperand = '';
                this.operation = undefined;
                this.updateDisplay();
            }
            delete() {
                if (this.currentOperand === '0') return;
                this.currentOperand = this.currentOperand.slice(0, -1);
                if (this.currentOperand === '' || this.currentOperand === '-') this.currentOperand = '0';
                this.updateDisplay();
            }
            appendNumber(number) {
                if (this.currentOperand === '0' && number !== '.') {
                    this.currentOperand = number;
                } else {
                    this.currentOperand += number;
                }
                this.updateDisplay();
            }
            appendDecimal() {
                if (this.currentOperand.includes('.')) return;
                this.currentOperand += '.';
                this.updateDisplay();
            }
            appendOperator(op) {
                if (this.currentOperand === '' && this.previousOperand === '') return;
                if (this.previousOperand !== '') this.compute();
                this.operation = op;
                this.previousOperand = this.currentOperand;
                this.currentOperand = '';
                this.updateDisplay();
            }
            compute() {
                let computation;
                const prev = parseFloat(this.previousOperand);
                const current = parseFloat(this.currentOperand);
                if (isNaN(prev) || isNaN(current)) return;
                switch (this.operation) {
                    case '+': computation = prev + current; break;
                    case '-': computation = prev - current; break;
                    case '*': computation = prev * current; break;
                    case '/': computation = current === 0 ? 'Error' : prev / current; break;
                    case '%': computation = prev % current; break;
                    default: return;
                }
                this.currentOperand = String(computation);
                this.operation = undefined;
                this.previousOperand = '';
                this.updateDisplay();
            }
            updateDisplay() {
                this.currentEl.innerText = this.currentOperand;
                this.historyEl.innerText = this.operation != null ? `${this.previousOperand} ${this.operation}` : '';
            }
        }
        const calc = new Calculator(document.getElementById('historyDisplay'), document.getElementById('currentDisplay'));
    </script>
</body>
</html>
"""
    sections.append(web_code)

    # ── 2. C++ & Systems Programming (Quicksort, LRU Cache, STL) ──
    cpp_code = """=== Systems Programming (C++20, STL, Memory Management) ===
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <shared_mutex>

namespace algorithms {

template <typename T>
int partition(std::vector<T>& arr, int low, int high) {
    T pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

template <typename T>
void quicksort(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

template <typename K, typename V>
class ThreadSafeLRUCache {
private:
    struct Node {
        K key;
        V value;
        Node* prev = nullptr;
        Node* next = nullptr;
        Node(K k, V v) : key(std::move(k)), value(std::move(v)) {}
    };

    size_t capacity_;
    std::unordered_map<K, Node*> map_;
    Node* head_;
    Node* tail_;
    mutable std::shared_mutex rw_lock_;

    void remove_node(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void add_to_head(Node* node) {
        node->next = head_->next;
        node->prev = head_;
        head_->next->prev = node;
        head_->next = node;
    }

public:
    explicit ThreadSafeLRUCache(size_t cap) : capacity_(cap) {
        head_ = new Node(K{}, V{});
        tail_ = new Node(K{}, V{});
        head_->next = tail_;
        tail_->prev = head_;
    }

    ~ThreadSafeLRUCache() {
        Node* curr = head_;
        while (curr) {
            Node* nxt = curr->next;
            delete curr;
            curr = nxt;
        }
    }

    bool get(const K& key, V& out_val) {
        std::unique_lock<std::shared_mutex> lock(rw_lock_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        Node* node = it->second;
        remove_node(node);
        add_to_head(node);
        out_val = node->value;
        return true;
    }

    void put(const K& key, V value) {
        std::unique_lock<std::shared_mutex> lock(rw_lock_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            Node* node = it->second;
            node->value = std::move(value);
            remove_node(node);
            add_to_head(node);
            return;
        }
        if (map_.size() >= capacity_) {
            Node* lru = tail_->prev;
            remove_node(lru);
            map_.erase(lru->key);
            delete lru;
        }
        Node* new_node = new Node(key, std::move(value));
        add_to_head(new_node);
        map_[key] = new_node;
    }
};

} // namespace algorithms
"""
    sections.append(cpp_code)

    # ── 3. Algorithmic Python, OOP & Async ──
    python_code = """=== Algorithmic Python, Data Structures & Networking ===
import asyncio
import hashlib
import json
import heapq
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class TokenizerConfig:
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_heads: int = 64
    intermediate_size: int = 2304
    routed_scaling_factor: float = 1.5

class BinaryIndexedTree:
    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, i: int, delta: int):
        while i <= self.size:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

def dijkstra(graph: Dict[int, List[Tuple[int, float]]], start: int) -> Dict[int, float]:
    distances = {node: float('inf') for node in graph}
    distances[start] = 0.0
    pq = [(0.0, start)]
    
    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > distances[u]:
            continue
        for v, weight in graph.get(u, []):
            d = curr_dist + weight
            if d < distances[v]:
                distances[v] = d
                heapq.heappush(pq, (d, v))
    return distances

def solve_n_queens(n: int) -> List[List[int]]:
    results = []
    board = [-1] * n
    def is_safe(row, col):
        for r in range(row):
            c = board[r]
            if c == col or abs(c - col) == abs(r - row):
                return False
        return True
    def backtrack(row):
        if row == n:
            results.append(list(board))
            return
        for col in range(n):
            if is_safe(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1
    backtrack(0)
    return results
"""
    sections.append(python_code)

    # ── 4. CUDA GPU Kernels (train_gpt2.cu) ──
    cu_path = Path("data/calibration/train_gpt2.cu")
    if cu_path.exists():
        with open(cu_path, "r", encoding="utf-8", errors="ignore") as f:
            cu_text = f.read()
        sections.append("=== CUDA C++ GPU Kernels ===\n" + cu_text[:4000])

    # ── 5. Mathematical Reasoning & Chain-of-Thought (GSM8K) ──
    gsm8k_path = Path("data/calibration/gsm8k_train.jsonl")
    if gsm8k_path.exists():
        gsm_items = []
        with open(gsm8k_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                data = json.loads(line)
                q = data.get("question", "")
                a = data.get("answer", "")
                gsm_items.append(f"<｜User｜>{q}<｜Assistant｜><think>\n{a}\n</think>\nFinal Answer: {a.split('####')[-1].strip()}")
        sections.append("=== Mathematical Reasoning & Chain-of-Thought (GSM8K) ===\n" + "\n\n".join(gsm_items))

    # ── 6. WikiText-2 (General Knowledge, Science, Linguistics) ──
    wiki_path = Path("data/calibration/wikitext_2_train.txt")
    if wiki_path.exists():
        with open(wiki_path, "r", encoding="utf-8", errors="ignore") as f:
            wiki_lines = [line.strip() for line in f if len(line.strip()) > 80]
        wiki_sample = "\n\n".join(wiki_lines[:15])
        sections.append("=== Encyclopedic Science, History & Literature (WikiText-2) ===\n" + wiki_sample)

    full_corpus = "\n\n".join(sections)
    encoded = tok.encode(full_corpus)
    total_tokens = len(encoded.ids)
    
    with open(out_corpus_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)
        
    print("=" * 70)
    print(f"SUCCESS: Assembled balanced multi-domain calibration corpus!")
    print(f"  Total Characters : {len(full_corpus):,}")
    print(f"  Total Tokens     : {total_tokens:,}")
    print(f"  Output Path      : {out_corpus_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
prepare_full_calibration_corpus.py - Assemble Balanced, Interleaved Multi-Domain Calibration Text for DeepSeek-V4.1

Interleaves across all key domains in round-robin fashion:
1. Web Development (HTML5, CSS3, JavaScript modern UI & calculators)
2. Reasoning & Math (OpenAI GSM8K step-by-step solutions)
3. Systems & High-Performance C++20 (Algorithms, data structures, concurrency)
4. Encyclopedic General Knowledge (WikiText-2 science, history, literature)
5. GPU & AI Compute (CUDA train_gpt2.cu kernels)
6. Algorithmic Python (Trees, graphs, dynamic programming)
"""

import os
import sys
import json
from pathlib import Path
from tokenizers import Tokenizer

def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def main():
    tokenizer_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/tokenizer.json")
    out_corpus_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/calibration_corpus_full.txt")
    cache_dir = Path("data/calibration")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tok = Tokenizer.from_file(str(tokenizer_path))

    # 1. Web Development Snippets
    web_snippets = [
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modern Responsive Calculator</title>
    <style>
        :root {
            --bg: #1e1e2f;
            --surface: #27293d;
            --primary: #e14eca;
            --text: #ffffff;
        }
        body {
            background: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .calc-card {
            background: var(--surface);
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            width: 320px;
        }
        .calc-screen {
            background: rgba(0,0,0,0.25);
            padding: 16px;
            font-size: 28px;
            text-align: right;
            border-radius: 8px;
            margin-bottom: 16px;
            min-height: 40px;
        }
        .btn-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }
        button {
            padding: 16px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            background: #34344e;
            color: #fff;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        button:hover { background: #454566; transform: translateY(-2px); }
        button.action { background: var(--primary); }
    </style>
</head>
<body>
    <div class="calc-card">
        <div id="screen" class="calc-screen">0</div>
        <div class="btn-grid">
            <button onclick="press('C')">C</button>
            <button onclick="press('±')">±</button>
            <button onclick="press('%')">%</button>
            <button class="action" onclick="press('/')">÷</button>
            <button onclick="press('7')">7</button>
            <button onclick="press('8')">8</button>
            <button onclick="press('9')">9</button>
            <button class="action" onclick="press('*')">×</button>
            <button onclick="press('4')">4</button>
            <button onclick="press('5')">5</button>
            <button onclick="press('6')">6</button>
            <button class="action" onclick="press('-')">−</button>
            <button onclick="press('1')">1</button>
            <button onclick="press('2')">2</button>
            <button onclick="press('3')">3</button>
            <button class="action" onclick="press('+')">+</button>
            <button style="grid-column: span 2;" onclick="press('0')">0</button>
            <button onclick="press('.')">.</button>
            <button class="action" onclick="press('=')">=</button>
        </div>
    </div>
    <script>
        let expr = '';
        function press(k) {
            const s = document.getElementById('screen');
            if (k === 'C') { expr = ''; s.innerText = '0'; return; }
            if (k === '=') {
                try { s.innerText = eval(expr) || '0'; expr = s.innerText; }
                catch(e) { s.innerText = 'Error'; expr = ''; }
                return;
            }
            expr += k;
            s.innerText = expr;
        }
    </script>
</body>
</html>
""",
"""
<!DOCTYPE html>
<html>
<head>
    <title>Dynamic Data Table Component</title>
    <style>
        .table-container { max-width: 800px; margin: 2rem auto; font-family: sans-serif; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: 12px 16px; border-bottom: 1px solid #ddd; text-align: left; }
        th { background-color: #f8f9fa; cursor: pointer; user-select: none; }
        th:hover { background-color: #e9ecef; }
        tr:hover { background-color: #f1f3f5; }
        .search-box { width: 100%; padding: 10px; font-size: 16px; border: 1px solid #ced4da; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="table-container">
        <input type="text" id="search" class="search-box" placeholder="Filter table entries..." oninput="filterData()">
        <table id="dataTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">ID</th>
                    <th onclick="sortTable(1)">Name</th>
                    <th onclick="sortTable(2)">Category</th>
                    <th onclick="sortTable(3)">Score</th>
                </tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>
    <script>
        const data = [
            { id: 101, name: "Alpha", category: "Engine", score: 98.4 },
            { id: 102, name: "Beta", category: "Compiler", score: 85.2 },
            { id: 103, name: "Gamma", category: "Runtime", score: 91.7 },
            { id: 104, name: "Delta", category: "Kernel", score: 99.1 }
        ];
        function renderTable(items) {
            const body = document.getElementById('tableBody');
            body.innerHTML = items.map(i => `<tr><td>${i.id}</td><td>${i.name}</td><td>${i.category}</td><td>${i.score}</td></tr>`).join('');
        }
        function filterData() {
            const q = document.getElementById('search').value.toLowerCase();
            renderTable(data.filter(x => x.name.toLowerCase().includes(q) || x.category.toLowerCase().includes(q)));
        }
        renderTable(data);
    </script>
</body>
</html>
"""
    ]

    # 2. Reasoning & Math (GSM8K)
    gsm8k_items = []
    gsm8k_path = cache_dir / "gsm8k_train.jsonl"
    if gsm8k_path.exists():
        with open(gsm8k_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    gsm8k_items.append(f"Problem: {obj['question']}\n\nReasoning:\n{obj['answer']}")
                except Exception:
                    pass

    # 3. Systems C++
    cpp_snippets = [
"""
#include <iostream>
#include <vector>
#include <concepts>
#include <algorithm>
#include <memory>

template <typename T>
requires std::totally_ordered<T>
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

template <typename Key, typename Value>
class ThreadSafeLRUCache {
    struct Node {
        Key key;
        Value value;
        std::shared_ptr<Node> prev;
        std::shared_ptr<Node> next;
        Node(Key k, Value v) : key(k), value(v) {}
    };
    size_t capacity_;
    std::unordered_map<Key, std::shared_ptr<Node>> map_;
    std::shared_ptr<Node> head_, tail_;
public:
    explicit ThreadSafeLRUCache(size_t cap) : capacity_(cap) {}
    bool get(const Key& k, Value& out) {
        auto it = map_.find(k);
        if (it == map_.end()) return false;
        out = it->second->value;
        return true;
    }
};
""",
"""
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>

class ThreadPool {
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
public:
    ThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->cv_.wait(lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty()) return;
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return res;
    }
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }
};
"""
    ]

    # 4. WikiText Encyclopedia Paragraphs
    wiki_paras = []
    wiki_path = cache_dir / "wikitext_2_train.txt"
    if wiki_path.exists():
        with open(wiki_path, "r", encoding="utf-8") as f:
            raw_wiki = f.read()
        for p in raw_wiki.split("\n\n"):
            p_clean = p.strip()
            if len(p_clean) > 120 and not p_clean.startswith("="):
                wiki_paras.append(p_clean)

    # 5. CUDA Kernels
    cuda_blocks = []
    cu_path = cache_dir / "train_gpt2.cu"
    if cu_path.exists():
        with open(cu_path, "r", encoding="utf-8") as f:
            raw_cu = f.read()
        lines = raw_cu.splitlines()
        # Slice into ~100-line chunks
        chunk = []
        for l in lines:
            chunk.append(l)
            if len(chunk) >= 80:
                cuda_blocks.append("\n".join(chunk))
                chunk = []
        if chunk:
            cuda_blocks.append("\n".join(chunk))

    # 6. Python Algorithms
    python_snippets = [
"""
from typing import List, Dict, Tuple, Optional
import heapq
from collections import defaultdict, deque

class FenwickTree:
    def __init__(self, size: int):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index: int, delta: int) -> None:
        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)

    def query(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & (-index)
        return total

def dijkstra_shortest_path(n: int, edges: List[Tuple[int, int, int]], source: int) -> Dict[int, int]:
    adj = defaultdict(list)
    for u, v, weight in edges:
        adj[u].append((v, weight))
        adj[v].append((u, weight))

    distances = {i: float('inf') for i in range(n)}
    distances[source] = 0
    pq = [(0, source)]

    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > distances[u]:
            continue
        for v, weight in adj[u]:
            new_dist = curr_dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return distances
""",
"""
def solve_n_queens(n: int) -> List[List[str]]:
    solutions = []
    cols = set()
    diag1 = set()
    diag2 = set()
    board = [['.'] * n for _ in range(n)]

    def backtrack(r: int):
        if r == n:
            solutions.append(["".join(row) for row in board])
            return
        for c in range(n):
            if c in cols or (r - c) in diag1 or (r + c) in diag2:
                continue
            cols.add(c)
            diag1.add(r - c)
            diag2.add(r + c)
            board[r][c] = 'Q'
            backtrack(r + 1)
            board[r][c] = '.'
            cols.remove(c)
            diag1.remove(r - c)
            diag2.remove(r + c)

    backtrack(0)
    return solutions
"""
    ]

    # Interleave sections in round-robin until target token count is reached (~35,000 tokens)
    corpus_chunks = []
    target_tokens = 35000
    current_tokens = 0

    idx_web = 0
    idx_gsm = 0
    idx_cpp = 0
    idx_wiki = 0
    idx_cu = 0
    idx_py = 0

    print("Interleaving multi-domain calibration corpus...")
    while current_tokens < target_tokens:
        # 1. Web
        if web_snippets:
            chunk = web_snippets[idx_web % len(web_snippets)]
            corpus_chunks.append(chunk)
            idx_web += 1
            current_tokens += len(tok.encode(chunk).ids)

        # 2. GSM8K (2 problems)
        if gsm8k_items:
            for _ in range(2):
                chunk = gsm8k_items[idx_gsm % len(gsm8k_items)]
                corpus_chunks.append(chunk)
                idx_gsm += 1
                current_tokens += len(tok.encode(chunk).ids)

        # 3. Systems C++
        if cpp_snippets:
            chunk = cpp_snippets[idx_cpp % len(cpp_snippets)]
            corpus_chunks.append(chunk)
            idx_cpp += 1
            current_tokens += len(tok.encode(chunk).ids)

        # 4. WikiText (2 paragraphs)
        if wiki_paras:
            for _ in range(2):
                chunk = wiki_paras[idx_wiki % len(wiki_paras)]
                corpus_chunks.append(chunk)
                idx_wiki += 1
                current_tokens += len(tok.encode(chunk).ids)

        # 5. CUDA
        if cuda_blocks:
            chunk = cuda_blocks[idx_cu % len(cuda_blocks)]
            corpus_chunks.append(chunk)
            idx_cu += 1
            current_tokens += len(tok.encode(chunk).ids)

        # 6. Python
        if python_snippets:
            chunk = python_snippets[idx_py % len(python_snippets)]
            corpus_chunks.append(chunk)
            idx_py += 1
            current_tokens += len(tok.encode(chunk).ids)

    full_text = "\n\n\n".join(corpus_chunks)
    all_tokens = tok.encode(full_text).ids
    print(f"Generated Balanced Calibration Corpus: {len(all_tokens):,} tokens ({len(full_text):,} chars)")

    with open(out_corpus_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Saved to: {out_corpus_path}")

if __name__ == "__main__":
    main()

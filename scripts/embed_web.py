import os

def file_to_c_array(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    lines = []
    chunk_size = 32
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        lines.append(', '.join(f'0x{b:02x}' for b in chunk))
    return f'// {len(data)} bytes\n' + ',\n'.join(lines), len(data)

def generate_header():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_dir = os.path.join(base_dir, 'web')
    out_path = os.path.join(base_dir, 'src', 'embedded_web.hpp')

    html_bytes, html_len = file_to_c_array(os.path.join(web_dir, 'index.html'))
    css_bytes, css_len = file_to_c_array(os.path.join(web_dir, 'style.css'))
    js_bytes, js_len = file_to_c_array(os.path.join(web_dir, 'script.js'))

    header = f'''#pragma once
#include <string_view>
#include <cstddef>

namespace moecher::embedded_web {{

inline const unsigned char INDEX_HTML_DATA[] = {{
{html_bytes}
}};

inline const unsigned char STYLE_CSS_DATA[] = {{
{css_bytes}
}};

inline const unsigned char SCRIPT_JS_DATA[] = {{
{js_bytes}
}};

inline std::string_view INDEX_HTML() {{
    return std::string_view(reinterpret_cast<const char*>(INDEX_HTML_DATA), sizeof(INDEX_HTML_DATA));
}}

inline std::string_view STYLE_CSS() {{
    return std::string_view(reinterpret_cast<const char*>(STYLE_CSS_DATA), sizeof(STYLE_CSS_DATA));
}}

inline std::string_view SCRIPT_JS() {{
    return std::string_view(reinterpret_cast<const char*>(SCRIPT_JS_DATA), sizeof(SCRIPT_JS_DATA));
}}

}} // namespace moecher::embedded_web
'''

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(header)
    print(f"Generated {out_path} successfully ({html_len} bytes HTML, {css_len} bytes CSS, {js_len} bytes JS)")

if __name__ == '__main__':
    generate_header()

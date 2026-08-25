import sys
content = open('src/server_single.cpp').read()
old_code = """        if (position > 0) {
            float t_loop = std::chrono::duration<float, std::milli>(loop_end - loop_start).count();
            printf("[PROFILE] t_entire_loop=%.3fms\\n", t_loop);
        }"""
new_code = """        if (position > 0) {
            float t_loop = std::chrono::duration<float, std::milli>(loop_end - loop_start).count();
            printf("[PROFILE] t_entire_loop=%.3fms\\n", t_loop);
        }"""
content = content.replace(old_code, new_code)
open('src/server_single.cpp', 'w').write(content)

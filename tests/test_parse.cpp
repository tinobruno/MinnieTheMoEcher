#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "third_party/json.hpp" // assuming json.hpp is available

using json = nlohmann::json;

int main() {
    std::string text = "<\xef\xbd\x9cDSML\xef\xbd\x9cinvoke name=\"get_weather\">\n<\xef\xbd\x9cDSML\xef\xbd\x9cparameter name=\"location\" string=\"true\">Tokyo</\xef\xbd\x9cDSML\xef\xbd\x9cparameter>\n</\xef\xbd\x9cDSML\xef\xbd\x9cinvoke>";
    
    std::string invoke_start = "<\xef\xbd\x9cDSML\xef\xbd\x9cinvoke name=\"";
    std::string param_start = "<\xef\xbd\x9cDSML\xef\xbd\x9cparameter name=\"";
    std::string param_end = "</\xef\xbd\x9cDSML\xef\xbd\x9cparameter>";
    std::string invoke_end = "</\xef\xbd\x9cDSML\xef\xbd\x9cinvoke>";
    
    size_t pos = 0;
    json tool_calls_json = json::array();
    int idx = 0;
    
    while ((pos = text.find(invoke_start, pos)) != std::string::npos) {
        pos += invoke_start.length();
        size_t name_end = text.find("\">", pos);
        if (name_end == std::string::npos) break;
        std::string func_name = text.substr(pos, name_end - pos);
        pos = name_end + 2;
        
        size_t inv_end = text.find(invoke_end, pos);
        if (inv_end == std::string::npos) inv_end = text.length();
        
        std::string args_str = "";
        json args_json = json::object();
        
        size_t ppos = pos;
        while ((ppos = text.find(param_start, ppos)) != std::string::npos && ppos < inv_end) {
            ppos += param_start.length();
            size_t pname_end = text.find("\"", ppos);
            if (pname_end == std::string::npos) break;
            std::string param_name = text.substr(ppos, pname_end - ppos);
            ppos = pname_end + 1;
            
            bool is_string = true;
            if (text.substr(ppos, 9) == " string=\"") {
                ppos += 9;
                size_t pstr_end = text.find("\">", ppos);
                if (pstr_end != std::string::npos) {
                    std::string str_val = text.substr(ppos, pstr_end - ppos);
                    is_string = (str_val == "true");
                    ppos = pstr_end + 2;
                }
            } else {
                size_t p_end = text.find(">", ppos);
                if (p_end != std::string::npos) ppos = p_end + 1;
            }
            
            size_t pend = text.find(param_end, ppos);
            if (pend == std::string::npos) break;
            
            std::string param_val = text.substr(ppos, pend - ppos);
            if (is_string) {
                args_json[param_name] = param_val;
            } else {
                try {
                    args_json[param_name] = json::parse(param_val);
                } catch(...) {
                    args_json[param_name] = param_val;
                }
            }
            ppos = pend + param_end.length();
        }
        
        tool_calls_json.push_back({
            {"index", idx++},
            {"id", "call_" + std::to_string(rand())},
            {"type", "function"},
            {"function", {
                {"name", func_name},
                {"arguments", args_json.dump()}
            }}
        });
        
        pos = inv_end + invoke_end.length();
    }
    
    std::cout << tool_calls_json.dump(4) << std::endl;
    return 0;
}

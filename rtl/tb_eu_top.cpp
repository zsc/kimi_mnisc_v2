//------------------------------------------------------------------------------
// MNISC EU Verilator Harness
// Drives EU RTL with program.bin and validates against AST simulator output
//------------------------------------------------------------------------------

#include <verilated.h>
#include <verilated_vcd_c.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <memory>

// Include Verilator generated header
// Note: Verilator generates Veu_top.h for module eu_top
#include "Veu_top.h"

// If JSONCPP is available, use it; otherwise use built-in simple JSON parser
#ifdef HAS_JSONCPP
#include <json/json.h>
#endif

// Default simulation parameters
#define DEFAULT_BUS_W 128
#define DEFAULT_INSN_W 32
#define BYTES_PER_BEAT (DEFAULT_BUS_W / 8)  // 16 bytes per beat

//------------------------------------------------------------------------------
// Global simulation state
//------------------------------------------------------------------------------
vluint64_t sim_time = 0;
bool trace_enabled = false;
bool backpressure_enabled = false;

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

// Load binary file into vector
std::vector<uint8_t> load_binary_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << path << std::endl;
        exit(1);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Error: Failed to read file: " << path << std::endl;
        exit(1);
    }
    
    return buffer;
}

// Load program.bin (u32 instruction stream, little-endian)
std::vector<uint32_t> load_program_bin(const char* path) {
    auto bytes = load_binary_file(path);
    
    if (bytes.size() % 4 != 0) {
        std::cerr << "Warning: program.bin size " << bytes.size() 
                  << " is not a multiple of 4 bytes" << std::endl;
    }
    
    std::vector<uint32_t> program;
    program.reserve(bytes.size() / 4);
    
    for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
        uint32_t word = bytes[i] | 
                       (static_cast<uint32_t>(bytes[i+1]) << 8) | 
                       (static_cast<uint32_t>(bytes[i+2]) << 16) | 
                       (static_cast<uint32_t>(bytes[i+3]) << 24);
        program.push_back(word);
    }
    
    return program;
}

//------------------------------------------------------------------------------
// Simple JSON Parser (header-only, minimal implementation)
//------------------------------------------------------------------------------
namespace simple_json {
    enum class ValueType { NULL_TYPE, BOOL, NUMBER, STRING, ARRAY, OBJECT };
    
    struct Value {
        ValueType type = ValueType::NULL_TYPE;
        bool bool_val = false;
        double num_val = 0;
        std::string str_val;
        std::vector<Value> array_val;
        std::map<std::string, Value> obj_val;
        
        Value() = default;
        Value(bool b) : type(ValueType::BOOL), bool_val(b) {}
        Value(double n) : type(ValueType::NUMBER), num_val(n) {}
        Value(int n) : type(ValueType::NUMBER), num_val(n) {}
        Value(size_t n) : type(ValueType::NUMBER), num_val(static_cast<double>(n)) {}
        Value(const std::string& s) : type(ValueType::STRING), str_val(s) {}
        Value(const char* s) : type(ValueType::STRING), str_val(s) {}
        
        bool isNull() const { return type == ValueType::NULL_TYPE; }
        bool isBool() const { return type == ValueType::BOOL; }
        bool isNumber() const { return type == ValueType::NUMBER; }
        bool isString() const { return type == ValueType::STRING; }
        bool isArray() const { return type == ValueType::ARRAY; }
        bool isObject() const { return type == ValueType::OBJECT; }
        
        bool asBool() const { return bool_val; }
        int asInt() const { return static_cast<int>(num_val); }
        uint32_t asUInt() const { return static_cast<uint32_t>(num_val); }
        size_t asUInt64() const { return static_cast<size_t>(num_val); }
        double asDouble() const { return num_val; }
        const std::string& asString() const { return str_val; }
        const std::vector<Value>& asArray() const { return array_val; }
        std::vector<Value>& asArray() { return array_val; }
        
        const Value& operator[](const std::string& key) const {
            auto it = obj_val.find(key);
            if (it != obj_val.end()) return it->second;
            static Value null_val;
            return null_val;
        }
        
        Value& operator[](const std::string& key) {
            return obj_val[key];
        }
        
        const Value& operator[](size_t idx) const {
            if (idx < array_val.size()) return array_val[idx];
            static Value null_val;
            return null_val;
        }
        
        Value& operator[](size_t idx) {
            if (idx >= array_val.size()) array_val.resize(idx + 1);
            return array_val[idx];
        }
        
        bool hasMember(const std::string& key) const {
            return obj_val.find(key) != obj_val.end();
        }
        
        std::vector<std::string> getMemberNames() const {
            std::vector<std::string> names;
            for (const auto& kv : obj_val) {
                names.push_back(kv.first);
            }
            return names;
        }
        
        // Push back for array
        void append(const Value& v) {
            type = ValueType::ARRAY;
            array_val.push_back(v);
        }
    };
    
    // Skip whitespace
    static const char* skip_ws(const char* p) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        return p;
    }
    
    // Forward declaration
    Value parse_value(const char*& p);
    
    // Parse string
    std::string parse_string(const char*& p) {
        if (*p != '"') return "";
        p++;
        std::string result;
        while (*p && *p != '"') {
            if (*p == '\\' && *(p+1)) {
                p++;
                switch (*p) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    default: result += *p; break;
                }
            } else {
                result += *p;
            }
            p++;
        }
        if (*p == '"') p++;
        return result;
    }
    
    // Parse number
    Value parse_number(const char*& p) {
        const char* start = p;
        if (*p == '-') p++;
        while (*p >= '0' && *p <= '9') p++;
        if (*p == '.') {
            p++;
            while (*p >= '0' && *p <= '9') p++;
        }
        if (*p == 'e' || *p == 'E') {
            p++;
            if (*p == '+' || *p == '-') p++;
            while (*p >= '0' && *p <= '9') p++;
        }
        double val = std::strtod(start, nullptr);
        return Value(val);
    }
    
    // Parse array
    Value parse_array(const char*& p) {
        Value arr;
        arr.type = ValueType::ARRAY;
        p++;
        p = skip_ws(p);
        if (*p == ']') { p++; return arr; }
        while (*p) {
            arr.array_val.push_back(parse_value(p));
            p = skip_ws(p);
            if (*p == ']') { p++; break; }
            if (*p == ',') p++;
            p = skip_ws(p);
        }
        return arr;
    }
    
    // Parse object
    Value parse_object(const char*& p) {
        Value obj;
        obj.type = ValueType::OBJECT;
        p++;
        p = skip_ws(p);
        if (*p == '}') { p++; return obj; }
        while (*p) {
            p = skip_ws(p);
            std::string key = parse_string(p);
            p = skip_ws(p);
            if (*p == ':') p++;
            p = skip_ws(p);
            obj.obj_val[key] = parse_value(p);
            p = skip_ws(p);
            if (*p == '}') { p++; break; }
            if (*p == ',') p++;
            p = skip_ws(p);
        }
        return obj;
    }
    
    // Parse value
    Value parse_value(const char*& p) {
        p = skip_ws(p);
        if (!*p) return Value();
        
        switch (*p) {
            case '"': return Value(parse_string(p));
            case '{': return parse_object(p);
            case '[': return parse_array(p);
            case 't': if (strncmp(p, "true", 4) == 0) { p += 4; return Value(true); } break;
            case 'f': if (strncmp(p, "false", 5) == 0) { p += 5; return Value(false); } break;
            case 'n': if (strncmp(p, "null", 4) == 0) { p += 4; return Value(); } break;
            case '-':
            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                return parse_number(p);
        }
        return Value();
    }
    
    // Parse from string
    Value parse(const char* json) {
        const char* p = json;
        return parse_value(p);
    }
    
    Value parse(const std::string& json) {
        return parse(json.c_str());
    }
    
    // Serialize to string (minimal implementation)
    void serialize(std::ostream& os, const Value& v, int indent = 0);
    
    void serialize_value(std::ostream& os, const Value& v, int indent) {
        switch (v.type) {
            case ValueType::NULL_TYPE: os << "null"; break;
            case ValueType::BOOL: os << (v.bool_val ? "true" : "false"); break;
            case ValueType::NUMBER: {
                if (v.num_val == static_cast<int64_t>(v.num_val)) {
                    os << static_cast<int64_t>(v.num_val);
                } else {
                    os << v.num_val;
                }
                break;
            }
            case ValueType::STRING: os << '"' << v.str_val << '"'; break;
            case ValueType::ARRAY: serialize(os, v, indent); break;
            case ValueType::OBJECT: serialize(os, v, indent); break;
        }
    }
    
    void serialize(std::ostream& os, const Value& v, int indent) {
        std::string ind(indent, ' ');
        if (v.isArray()) {
            os << "[\n";
            for (size_t i = 0; i < v.array_val.size(); i++) {
                os << ind << "  ";
                serialize_value(os, v.array_val[i], indent + 2);
                if (i + 1 < v.array_val.size()) os << ",";
                os << "\n";
            }
            os << ind << "]";
        } else if (v.isObject()) {
            os << "{\n";
            bool first = true;
            for (const auto& kv : v.obj_val) {
                if (!first) os << ",\n";
                first = false;
                os << ind << "  " << '"' << kv.first << "\": ";
                serialize_value(os, kv.second, indent + 2);
            }
            os << "\n" << ind << "}";
        } else {
            serialize_value(os, v, indent);
        }
    }
}

// Load program_meta.json
simple_json::Value load_program_meta(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open meta file: " << path << std::endl;
        // Return empty object instead of error
        return simple_json::Value();
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    return simple_json::parse(buffer.str());
}

//------------------------------------------------------------------------------
// Simple Safetensors Parser
//------------------------------------------------------------------------------
namespace simple_safetensors {
    struct TensorInfo {
        std::vector<uint8_t> data;
        std::vector<size_t> shape;
        std::string dtype;
    };
    
    std::map<std::string, TensorInfo> load(const char* path) {
        std::map<std::string, TensorInfo> tensors;
        
        auto file_data = load_binary_file(path);
        if (file_data.size() < 8) {
            std::cerr << "Warning: safetensors file too small: " << path << std::endl;
            return tensors;
        }
        
        // Read header size (first 8 bytes, little-endian uint64)
        uint64_t header_size = 0;
        for (int i = 0; i < 8; i++) {
            header_size |= (static_cast<uint64_t>(file_data[i]) << (8 * i));
        }
        
        if (8 + header_size > file_data.size()) {
            std::cerr << "Warning: Invalid header size in " << path << std::endl;
            return tensors;
        }
        
        // Parse JSON header
        std::string header_json(reinterpret_cast<char*>(file_data.data()) + 8, header_size);
        auto header = simple_json::parse(header_json);
        
        if (!header.isObject()) {
            std::cerr << "Warning: Invalid safetensors header in " << path << std::endl;
            return tensors;
        }
        
        // Parse each tensor
        for (const auto& key : header.getMemberNames()) {
            if (key == "__metadata__") continue;
            
            auto tensor_info = header[key];
            if (!tensor_info.isObject()) continue;
            
            TensorInfo info;
            
            if (tensor_info.hasMember("dtype")) {
                info.dtype = tensor_info["dtype"].asString();
            }
            
            if (tensor_info.hasMember("shape") && tensor_info["shape"].isArray()) {
                for (const auto& dim : tensor_info["shape"].asArray()) {
                    info.shape.push_back(dim.asUInt64());
                }
            }
            
            if (tensor_info.hasMember("data_offsets") && tensor_info["data_offsets"].isArray()) {
                auto offsets = tensor_info["data_offsets"].asArray();
                if (offsets.size() == 2) {
                    size_t start_offset = 8 + header_size + offsets[0].asUInt64();
                    size_t end_offset = 8 + header_size + offsets[1].asUInt64();
                    
                    if (end_offset <= file_data.size() && start_offset < end_offset) {
                        info.data.assign(file_data.begin() + start_offset, 
                                        file_data.begin() + end_offset);
                    }
                }
            }
            
            tensors[key] = std::move(info);
        }
        
        return tensors;
    }
    
    // Save tensors to safetensors file
    bool save(const char* path, const std::map<std::string, TensorInfo>& tensors) {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create output file: " << path << std::endl;
            return false;
        }
        
        // Build JSON header
        simple_json::Value header;
        header.type = simple_json::ValueType::OBJECT;
        
        size_t data_offset = 0;
        std::vector<uint8_t> all_data;
        
        for (const auto& kv : tensors) {
            const std::string& name = kv.first;
            const TensorInfo& info = kv.second;
            
            simple_json::Value tensor_obj;
            tensor_obj.type = simple_json::ValueType::OBJECT;
            tensor_obj["dtype"] = simple_json::Value(info.dtype);
            
            simple_json::Value shape_arr;
            shape_arr.type = simple_json::ValueType::ARRAY;
            for (auto dim : info.shape) {
                shape_arr.append(simple_json::Value(static_cast<double>(dim)));
            }
            tensor_obj["shape"] = shape_arr;
            
            simple_json::Value offsets_arr;
            offsets_arr.type = simple_json::ValueType::ARRAY;
            offsets_arr.append(simple_json::Value(static_cast<double>(data_offset)));
            offsets_arr.append(simple_json::Value(static_cast<double>(data_offset + info.data.size())));
            tensor_obj["data_offsets"] = offsets_arr;
            
            header.obj_val[name] = tensor_obj;
            
            data_offset += info.data.size();
            all_data.insert(all_data.end(), info.data.begin(), info.data.end());
        }
        
        // Serialize header
        std::ostringstream header_ss;
        simple_json::serialize(header_ss, header);
        std::string header_str = header_ss.str();
        
        // Pad header to 8-byte boundary
        while (header_str.size() % 8 != 0) {
            header_str += ' ';
        }
        
        uint64_t header_size = header_str.size();
        
        // Write header size (8 bytes, little-endian)
        for (int i = 0; i < 8; i++) {
            file.put(static_cast<char>((header_size >> (8 * i)) & 0xFF));
        }
        
        // Write header
        file.write(header_str.c_str(), header_str.size());
        
        // Write data
        file.write(reinterpret_cast<const char*>(all_data.data()), all_data.size());
        
        file.close();
        return true;
    }
}

//------------------------------------------------------------------------------
// Wide Data Bus Helpers (for 128-bit bus in Verilator)
//------------------------------------------------------------------------------

// Pack bytes into Verilator's VlWide signal format
// For 128-bit bus, Verilator uses VlWide<4> (4 x 32-bit words)
void pack_to_wide_signal(Veu_top* top, const uint8_t* data, size_t num_bytes, bool is_weight) {
    // For 128-bit signals, Verilator uses VlWide<4> which is 4 x 32-bit words
    // Access via .data() to get the underlying array
    
    // Clear the signal first
    if (is_weight) {
        for (int i = 0; i < 4; i++) {
            top->wgt_in_data.m_storage[i] = 0;
        }
        // Copy bytes (little-endian)
        for (size_t i = 0; i < num_bytes && i < 16; i++) {
            int word_idx = i / 4;
            int byte_idx = i % 4;
            top->wgt_in_data.m_storage[word_idx] |= 
                static_cast<uint32_t>(data[i]) << (8 * byte_idx);
        }
    } else {
        for (int i = 0; i < 4; i++) {
            top->act_in_data.m_storage[i] = 0;
        }
        for (size_t i = 0; i < num_bytes && i < 16; i++) {
            int word_idx = i / 4;
            int byte_idx = i % 4;
            top->act_in_data.m_storage[word_idx] |= 
                static_cast<uint32_t>(data[i]) << (8 * byte_idx);
        }
    }
}

// Unpack wide signal to bytes
void unpack_from_wide_signal(Veu_top* top, uint8_t* data, size_t num_bytes) {
    // Extract from VlWide<4> (4 x 32-bit words)
    for (size_t i = 0; i < num_bytes && i < 16; i++) {
        int word_idx = i / 4;
        int byte_idx = i % 4;
        data[i] = static_cast<uint8_t>((top->out_data.m_storage[word_idx] >> (8 * byte_idx)) & 0xFF);
    }
}

//------------------------------------------------------------------------------
// Stream Driver Class
//------------------------------------------------------------------------------

class StreamDriver {
public:
    StreamDriver(Veu_top* top, VerilatedContext* contextp, VerilatedVcdC* tfp = nullptr) 
        : top_(top), contextp_(contextp), tfp_(tfp), cycle_count_(0) {}
    
    // Clock tick with optional tracing
    void tick() {
        top_->clk = 0;
        contextp_->timeInc(1);
        top_->eval();
        if (tfp_) tfp_->dump(contextp_->time());
        
        top_->clk = 1;
        contextp_->timeInc(1);
        top_->eval();
        if (tfp_) tfp_->dump(contextp_->time());
        
        cycle_count_++;
    }
    
    // Reset EU
    void reset(int cycles = 10) {
        std::cout << "[Harness] Resetting EU for " << cycles << " cycles..." << std::endl;
        
        top_->rst_n = 0;
        top_->insn_valid = 0;
        top_->insn_data = 0;
        top_->wgt_in_valid = 0;
        top_->act_in_valid = 0;
        top_->out_ready = 1;
        
        // Clear wide data signals
        for (int i = 0; i < 4; i++) {
            top_->wgt_in_data.m_storage[i] = 0;
            top_->act_in_data.m_storage[i] = 0;
        }
        
        for (int i = 0; i < cycles; i++) {
            tick();
        }
        
        top_->rst_n = 1;
        std::cout << "[Harness] Reset complete" << std::endl;
    }
    
    // Send instruction word with ready/valid handshake
    bool send_insn(uint32_t insn_word, int timeout_cycles = 1000) {
        top_->insn_data = insn_word;
        top_->insn_valid = 1;
        
        int cycles = 0;
        while (cycles < timeout_cycles) {
            tick();
            if (top_->insn_ready) {
                top_->insn_valid = 0;
                return true;
            }
            cycles++;
        }
        
        std::cerr << "Error: Instruction send timeout at cycle " << cycle_count_ << std::endl;
        top_->insn_valid = 0;
        return false;
    }
    
    // Send entire program
    bool send_program(const std::vector<uint32_t>& program) {
        std::cout << "[Harness] Sending program with " << program.size() << " words" << std::endl;
        
        for (size_t i = 0; i < program.size(); i++) {
            if (!send_insn(program[i])) {
                std::cerr << "Error: Failed to send instruction " << i 
                          << " (0x" << std::hex << program[i] << std::dec << ")" << std::endl;
                return false;
            }
            
            if ((i + 1) % 100 == 0 || i == program.size() - 1) {
                std::cout << "[Harness] Sent " << (i + 1) << "/" << program.size() 
                          << " instructions" << std::endl;
            }
        }
        
        std::cout << "[Harness] Program sent successfully" << std::endl;
        return true;
    }
    
    // Send data via stream interface
    void send_stream_data(const uint8_t* data, size_t num_bytes, bool is_weight) {
        size_t bytes_sent = 0;
        uint8_t beat_data[BYTES_PER_BEAT] = {0};
        
        std::cout << "[Harness] Sending " << num_bytes << " " 
                  << (is_weight ? "weight" : "activation") << " bytes" << std::endl;
        
        while (bytes_sent < num_bytes) {
            // Prepare beat data
            size_t bytes_in_beat = std::min(static_cast<size_t>(BYTES_PER_BEAT), num_bytes - bytes_sent);
            std::memcpy(beat_data, data + bytes_sent, bytes_in_beat);
            
            // Pack and set data
            pack_to_wide_signal(top_, beat_data, bytes_in_beat, is_weight);
            
            if (is_weight) {
                top_->wgt_in_valid = 1;
            } else {
                top_->act_in_valid = 1;
            }
            
            // Wait for ready
            int wait_cycles = 0;
            while ((is_weight ? !top_->wgt_in_ready : !top_->act_in_ready) && 
                   wait_cycles < 1000) {
                tick();
                wait_cycles++;
            }
            
            if (wait_cycles >= 1000) {
                std::cerr << "Error: Stream ready timeout at byte " << bytes_sent << std::endl;
                break;
            }
            
            tick();
            bytes_sent += bytes_in_beat;
        }
        
        if (is_weight) {
            top_->wgt_in_valid = 0;
        } else {
            top_->act_in_valid = 0;
        }
        
        std::cout << "[Harness] Sent " << bytes_sent << " bytes in " 
                  << (bytes_sent + BYTES_PER_BEAT - 1) / BYTES_PER_BEAT << " beats" << std::endl;
    }
    
    // Convenience methods
    void send_weight_bytes(const uint8_t* data, size_t num_bytes) {
        send_stream_data(data, num_bytes, true);
    }
    
    void send_activation_bytes(const uint8_t* data, size_t num_bytes) {
        send_stream_data(data, num_bytes, false);
    }
    
    // Receive output bytes
    std::vector<uint8_t> receive_output_bytes(size_t num_bytes) {
        std::vector<uint8_t> received;
        received.reserve(num_bytes);
        
        std::cout << "[Harness] Receiving " << num_bytes << " output bytes" << std::endl;
        
        size_t bytes_received = 0;
        uint8_t beat_data[BYTES_PER_BEAT];
        int timeout_cycles = 0;
        const int MAX_TIMEOUT = 10000;
        
        while (bytes_received < num_bytes && timeout_cycles < MAX_TIMEOUT) {
            // Apply backpressure if enabled
            top_->out_ready = backpressure_enabled ? 
                !apply_backpressure(cycle_count_) : 1;
            
            tick();
            timeout_cycles++;
            
            if (top_->out_valid && top_->out_ready) {
                // Unpack received data
                unpack_from_wide_signal(top_, beat_data, BYTES_PER_BEAT);
                
                size_t bytes_to_copy = std::min(static_cast<size_t>(BYTES_PER_BEAT), num_bytes - bytes_received);
                received.insert(received.end(), beat_data, beat_data + bytes_to_copy);
                bytes_received += bytes_to_copy;
                timeout_cycles = 0;
            }
            
            // Check for errors
            if (top_->error_valid) {
                std::cerr << "Error: EU reported error code " << top_->error_code 
                          << " at cycle " << cycle_count_ << std::endl;
                break;
            }
        }
        
        top_->out_ready = 1;
        
        if (timeout_cycles >= MAX_TIMEOUT) {
            std::cerr << "Error: Output receive timeout" << std::endl;
        }
        
        std::cout << "[Harness] Received " << received.size() << " output bytes" << std::endl;
        return received;
    }
    
    // Enable/disable backpressure testing
    void set_backpressure_mode(bool enable) {
        backpressure_enabled = enable;
        std::cout << "[Harness] Backpressure testing: " 
                  << (enable ? "enabled" : "disabled") << std::endl;
    }
    
    // Get current cycle count
    vluint64_t get_cycle_count() const {
        return cycle_count_;
    }
    
    // Wait for done signal
    bool wait_for_done(int timeout_cycles = 1000000) {
        std::cout << "[Harness] Waiting for done signal..." << std::endl;
        
        int cycles = 0;
        while (cycles < timeout_cycles) {
            tick();
            
            if (top_->error_valid) {
                std::cerr << "Error: EU reported error code " << top_->error_code 
                          << " at cycle " << cycle_count_ << std::endl;
                return false;
            }
            
            if (top_->done) {
                std::cout << "[Harness] EU done at cycle " << cycle_count_ << std::endl;
                return true;
            }
            cycles++;
        }
        
        std::cerr << "Error: Timeout waiting for done after " << timeout_cycles 
                  << " cycles" << std::endl;
        return false;
    }
    
    // Run cycles
    void run_cycles(int cycles) {
        for (int i = 0; i < cycles; i++) {
            tick();
        }
    }

private:
    Veu_top* top_;
    VerilatedContext* contextp_;
    VerilatedVcdC* tfp_;
    vluint64_t cycle_count_;
    bool backpressure_enabled = false;
    
    // Periodic backpressure: every 17 cycles, pull low for 5 cycles
    bool apply_backpressure(vluint64_t cycle) {
        return (cycle % 17) < 5;  // Low for 5 cycles, high for 12
    }
};

//------------------------------------------------------------------------------
// Output Collector
//------------------------------------------------------------------------------

class OutputCollector {
public:
    void save_tensor(const std::string& tensor_id, 
                     const std::vector<uint8_t>& data,
                     const std::vector<size_t>& shape,
                     const std::string& dtype = "U8") {
        simple_safetensors::TensorInfo info;
        info.data = data;
        info.shape = shape;
        info.dtype = dtype;
        tensors_[tensor_id] = std::move(info);
        
        std::cout << "[Harness] Collected tensor " << tensor_id 
                  << " with " << data.size() << " bytes, shape [";
        for (size_t i = 0; i < shape.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << shape[i];
        }
        std::cout << "]" << std::endl;
    }
    
    bool save_to_safetensors(const char* path) {
        std::cout << "[Harness] Saving " << tensors_.size() 
                  << " tensors to " << path << std::endl;
        return simple_safetensors::save(path, tensors_);
    }
    
    size_t num_tensors() const {
        return tensors_.size();
    }

private:
    std::map<std::string, simple_safetensors::TensorInfo> tensors_;
};

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  +program=<file>    Program binary file (default: program.bin)" << std::endl;
    std::cout << "  +meta=<file>       Program metadata JSON (default: program_meta.json)" << std::endl;
    std::cout << "  +model=<file>      Model weights safetensors (default: model.safetensors)" << std::endl;
    std::cout << "  +input=<file>      Input activations safetensors (default: input.safetensors)" << std::endl;
    std::cout << "  +output=<file>     Output safetensors file (default: verilator_output.safetensors)" << std::endl;
    std::cout << "  +trace             Enable VCD waveform tracing" << std::endl;
    std::cout << "  +backpressure      Enable backpressure testing" << std::endl;
    std::cout << "  +help              Print this help message" << std::endl;
}

int main(int argc, char** argv) {
    // Default file paths
    const char* program_bin = "program.bin";
    const char* program_meta = "program_meta.json";
    const char* model_safetensors = "model.safetensors";
    const char* input_safetensors = "input.safetensors";
    const char* output_safetensors = "verilator_output.safetensors";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "+help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.substr(0, 9) == "+program=") {
            program_bin = argv[i] + 9;
        } else if (arg.substr(0, 6) == "+meta=") {
            program_meta = argv[i] + 6;
        } else if (arg.substr(0, 7) == "+model=") {
            model_safetensors = argv[i] + 7;
        } else if (arg.substr(0, 7) == "+input=") {
            input_safetensors = argv[i] + 7;
        } else if (arg.substr(0, 8) == "+output=") {
            output_safetensors = argv[i] + 8;
        } else if (arg == "+trace") {
            trace_enabled = true;
        } else if (arg == "+backpressure") {
            backpressure_enabled = true;
        }
    }
    
    std::cout << "=================================================" << std::endl;
    std::cout << "MNISC EU Verilator Harness" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Program: " << program_bin << std::endl;
    std::cout << "Meta:    " << program_meta << std::endl;
    std::cout << "Model:   " << model_safetensors << std::endl;
    std::cout << "Input:   " << input_safetensors << std::endl;
    std::cout << "Output:  " << output_safetensors << std::endl;
    std::cout << "Trace:   " << (trace_enabled ? "enabled" : "disabled") << std::endl;
    std::cout << "Backpressure: " << (backpressure_enabled ? "enabled" : "disabled") << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Load input files
    std::cout << "[Harness] Loading input files..." << std::endl;
    
    auto program = load_program_bin(program_bin);
    auto meta = load_program_meta(program_meta);
    auto model_tensors = simple_safetensors::load(model_safetensors);
    auto input_tensors = simple_safetensors::load(input_safetensors);
    
    std::cout << "[Harness] Loaded " << program.size() << " program words" << std::endl;
    std::cout << "[Harness] Loaded " << model_tensors.size() << " model tensors" << std::endl;
    std::cout << "[Harness] Loaded " << input_tensors.size() << " input tensors" << std::endl;
    
    // Initialize Verilator
    VerilatedContext* contextp = new VerilatedContext;
    contextp->commandArgs(argc, argv);
    
    std::cout << "[Harness] Creating EU top module..." << std::endl;
    Veu_top* top = new Veu_top(contextp);
    
    // Setup tracing
    VerilatedVcdC* tfp = nullptr;
    if (trace_enabled) {
        std::cout << "[Harness] Enabling VCD trace to eu_top.vcd" << std::endl;
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        top->trace(tfp, 99);
        tfp->open("eu_top.vcd");
    }
    
    // Create driver and collector
    StreamDriver driver(top, contextp, tfp);
    driver.set_backpressure_mode(backpressure_enabled);
    OutputCollector collector;
    
    // Main execution
    std::cout << "[Harness] Starting simulation..." << std::endl;
    
    // 1. Reset EU
    driver.reset(20);
    
    // 2. Send program
    if (!driver.send_program(program)) {
        std::cerr << "Error: Failed to send program" << std::endl;
        return 1;
    }
    
    // 3. Execute according to metadata
    if (meta.hasMember("instructions")) {
        const auto& instructions = meta["instructions"].asArray();
        std::cout << "[Harness] Executing " << instructions.size() << " instructions from metadata" << std::endl;
        
        for (size_t i = 0; i < instructions.size(); i++) {
            const auto& instr = instructions[i];
            std::string opcode = instr.hasMember("opcode") ? instr["opcode"].asString() : "UNKNOWN";
            
            std::cout << "[Harness] [" << i << "] Executing: " << opcode << std::endl;
            
            if (opcode == "CONV3X3" || opcode == "GEMM") {
                // Send weights
                if (instr.hasMember("weight")) {
                    auto weight_meta = instr["weight"];
                    std::string wgt_key = weight_meta.hasMember("tensor_id") ? 
                        weight_meta["tensor_id"].asString() : "";
                    size_t wgt_bytes = weight_meta.hasMember("bytes") ? 
                        weight_meta["bytes"].asUInt64() : 0;
                    
                    auto it = model_tensors.find(wgt_key);
                    if (it != model_tensors.end() && wgt_bytes > 0) {
                        size_t send_bytes = std::min(wgt_bytes, it->second.data.size());
                        driver.send_weight_bytes(it->second.data.data(), send_bytes);
                    } else if (wgt_bytes > 0) {
                        // Send zeros as placeholder
                        std::vector<uint8_t> zeros(wgt_bytes, 0);
                        driver.send_weight_bytes(zeros.data(), wgt_bytes);
                    }
                }
                
                // Send activations
                if (instr.hasMember("activation")) {
                    auto act_meta = instr["activation"];
                    std::string act_key = act_meta.hasMember("tensor_id") ? 
                        act_meta["tensor_id"].asString() : "";
                    size_t act_bytes = act_meta.hasMember("bytes") ? 
                        act_meta["bytes"].asUInt64() : 0;
                    
                    auto it = input_tensors.find(act_key);
                    if (it != input_tensors.end() && act_bytes > 0) {
                        size_t send_bytes = std::min(act_bytes, it->second.data.size());
                        driver.send_activation_bytes(it->second.data.data(), send_bytes);
                    } else {
                        // Try model tensors as fallback
                        it = model_tensors.find(act_key);
                        if (it != model_tensors.end() && act_bytes > 0) {
                            size_t send_bytes = std::min(act_bytes, it->second.data.size());
                            driver.send_activation_bytes(it->second.data.data(), send_bytes);
                        } else if (act_bytes > 0) {
                            std::vector<uint8_t> zeros(act_bytes, 0);
                            driver.send_activation_bytes(zeros.data(), act_bytes);
                        }
                    }
                }
                
                // Receive output
                if (instr.hasMember("output")) {
                    auto out_meta = instr["output"];
                    size_t out_bytes = out_meta.hasMember("bytes") ? 
                        out_meta["bytes"].asUInt64() : 0;
                    std::string out_id = out_meta.hasMember("tensor_id") ? 
                        out_meta["tensor_id"].asString() : ("output_" + std::to_string(i));
                    
                    if (out_bytes > 0) {
                        auto out_data = driver.receive_output_bytes(out_bytes);
                        
                        std::vector<size_t> shape;
                        if (out_meta.hasMember("shape")) {
                            for (const auto& dim : out_meta["shape"].asArray()) {
                                shape.push_back(dim.asUInt64());
                            }
                        }
                        
                        collector.save_tensor(out_id, out_data, shape);
                    }
                }
            }
            else if (opcode == "POOL2D" || opcode == "UNPOOL2D" || opcode == "ACT_QUANT") {
                // Similar handling for single-input ops
                if (instr.hasMember("activation")) {
                    auto act_meta = instr["activation"];
                    size_t act_bytes = act_meta.hasMember("bytes") ? 
                        act_meta["bytes"].asUInt64() : 0;
                    
                    if (act_bytes > 0) {
                        std::vector<uint8_t> zeros(act_bytes, 0);
                        driver.send_activation_bytes(zeros.data(), act_bytes);
                    }
                }
                
                if (instr.hasMember("output")) {
                    auto out_meta = instr["output"];
                    size_t out_bytes = out_meta.hasMember("bytes") ? 
                        out_meta["bytes"].asUInt64() : 0;
                    
                    if (out_bytes > 0) {
                        auto out_data = driver.receive_output_bytes(out_bytes);
                        std::string out_id = out_meta.hasMember("tensor_id") ? 
                            out_meta["tensor_id"].asString() : ("output_" + std::to_string(i));
                        
                        std::vector<size_t> shape;
                        if (out_meta.hasMember("shape")) {
                            for (const auto& dim : out_meta["shape"].asArray()) {
                                shape.push_back(dim.asUInt64());
                            }
                        }
                        
                        collector.save_tensor(out_id, out_data, shape);
                    }
                }
            }
            else if (opcode == "CONCAT_C") {
                // Handle concat - would need to send two activation streams
                // For now, simplified handling
            }
            else if (opcode == "OPC_END" || opcode == "END") {
                std::cout << "[Harness] Reached end of program" << std::endl;
                break;
            }
        }
    } else {
        std::cout << "[Harness] No metadata found, running basic test..." << std::endl;
        // Run basic test - just wait for done
        driver.run_cycles(100);
    }
    
    // 4. Wait for completion
    if (!driver.wait_for_done()) {
        std::cerr << "Error: Simulation did not complete successfully" << std::endl;
    }
    
    std::cout << "[Harness] Simulation completed at cycle " << driver.get_cycle_count() << std::endl;
    
    // 5. Save outputs
    if (collector.num_tensors() > 0) {
        if (!collector.save_to_safetensors(output_safetensors)) {
            std::cerr << "Error: Failed to save output" << std::endl;
        }
    } else {
        std::cout << "[Harness] No output tensors collected" << std::endl;
    }
    
    // Cleanup
    if (tfp) {
        tfp->close();
        delete tfp;
    }
    delete top;
    delete contextp;
    
    std::cout << "=================================================" << std::endl;
    std::cout << "[Harness] Simulation finished successfully" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    return 0;
}

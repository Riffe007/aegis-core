#pragma once
// A minimal JSON reader for the documents Core itself writes (arch/sm_<arch>.json).
// Values: null, bool, number (double), string, array, object. Malformed input
// yields a null root and ok() == false; no exceptions, no allocation limits
// beyond the document. Escapes: \" \\ \/ \b \f \n \r \t; \uXXXX is kept as text
// (no arch file uses it).
#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace aegis::hw::json {

struct Value {
  enum class Kind { Null, Bool, Number, String, Array, Object } kind = Kind::Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::map<std::string, Value> obj;

  // Member lookup; a null value when absent or not an object.
  const Value& operator[](const std::string& key) const {
    static const Value none;
    if (kind != Kind::Object) return none;
    const auto it = obj.find(key);
    return it == obj.end() ? none : it->second;
  }
  bool is_number() const { return kind == Kind::Number; }
  bool is_string() const { return kind == Kind::String; }
  double number_or(double d) const { return is_number() ? num : d; }
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}
  bool parse(Value* out) {
    ws();
    if (!value(out, 0)) return false;
    ws();
    return i_ == s_.size();
  }

 private:
  const std::string& s_;
  size_t i_ = 0;
  static constexpr int kMaxDepth = 64;

  void ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  bool lit(const char* w) {
    const size_t n = std::char_traits<char>::length(w);
    if (s_.compare(i_, n, w) != 0) return false;
    i_ += n;
    return true;
  }
  bool string(std::string* out) {
    if (i_ >= s_.size() || s_[i_] != '"') return false;
    ++i_;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) return false;
        const char e = s_[i_++];
        switch (e) {
          case '"': c = '"'; break;
          case '\\': c = '\\'; break;
          case '/': c = '/'; break;
          case 'b': c = '\b'; break;
          case 'f': c = '\f'; break;
          case 'n': c = '\n'; break;
          case 'r': c = '\r'; break;
          case 't': c = '\t'; break;
          case 'u': out->append("\\u"); continue;
          default: return false;
        }
      }
      out->push_back(c);
    }
    if (i_ >= s_.size()) return false;
    ++i_;
    return true;
  }
  bool value(Value* v, int depth) {
    if (depth > kMaxDepth || i_ >= s_.size()) return false;
    const char c = s_[i_];
    if (c == '{') {
      v->kind = Value::Kind::Object;
      ++i_;
      ws();
      if (i_ < s_.size() && s_[i_] == '}') return ++i_, true;
      for (;;) {
        ws();
        std::string key;
        if (!string(&key)) return false;
        ws();
        if (i_ >= s_.size() || s_[i_++] != ':') return false;
        ws();
        if (!value(&v->obj[key], depth + 1)) return false;
        ws();
        if (i_ >= s_.size()) return false;
        if (s_[i_] == ',') { ++i_; continue; }
        if (s_[i_] == '}') return ++i_, true;
        return false;
      }
    }
    if (c == '[') {
      v->kind = Value::Kind::Array;
      ++i_;
      ws();
      if (i_ < s_.size() && s_[i_] == ']') return ++i_, true;
      for (;;) {
        ws();
        v->arr.emplace_back();
        if (!value(&v->arr.back(), depth + 1)) return false;
        ws();
        if (i_ >= s_.size()) return false;
        if (s_[i_] == ',') { ++i_; continue; }
        if (s_[i_] == ']') return ++i_, true;
        return false;
      }
    }
    if (c == '"') {
      v->kind = Value::Kind::String;
      return string(&v->str);
    }
    if (lit("true")) { v->kind = Value::Kind::Bool; v->b = true; return true; }
    if (lit("false")) { v->kind = Value::Kind::Bool; return true; }
    if (lit("null")) return true;
    const char* begin = s_.c_str() + i_;
    char* end = nullptr;
    const double d = std::strtod(begin, &end);
    if (end == begin) return false;
    i_ += size_t(end - begin);
    v->kind = Value::Kind::Number;
    v->num = d;
    return true;
  }
};

// Parses `text`; false (and *out null) if it is not one JSON value.
inline bool parse(const std::string& text, Value* out) {
  *out = Value{};
  Parser p(text);
  if (!p.parse(out)) {
    *out = Value{};
    return false;
  }
  return true;
}

}  // namespace aegis::hw::json

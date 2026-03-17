#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct OperatorChoice {
  std::string partition;
  long long opcost = 0;
};

struct TensorLayoutUse {
  std::string tensor;
  std::string layout;
  bool is_producer = false;
};

struct TransposeCost {
  std::string tensor;
  std::string src_layout;
  std::string dst_layout;
  long long tcost = 0;
};

struct ParsedInstance {
  std::unordered_map<std::string, std::vector<OperatorChoice>> St;
  std::unordered_map<std::string, std::vector<TensorLayoutUse>> choice_uses;
  std::vector<TransposeCost> transpose_costs;
  std::vector<std::pair<std::string, std::string>> edges;
};

enum class VarType { X, P, A };

struct Variable {
  VarType type;
  std::string op;
  std::string tensor;
  std::string layout;
  int domain_size = 0;
};

struct InteractionGraph {
  std::vector<Variable> vars;
  std::set<std::pair<int, int>> E;
};

struct TD {
  int n_bags_declared = 0;
  int width_plus_one_declared = 0;
  int n_graph_vertices = 0;
  std::map<int, std::set<int>> bags;  // bag_id -> vertices
  std::vector<std::pair<int, int>> edges;
};

struct RootedTD {
  int root = 0;
  std::vector<std::set<int>> bag;      // 1-based, bag[0] unused
  std::vector<std::vector<int>> ch;    // children
};

struct NiceCheckResult {
  bool ok = false;
  int node = -1;
  std::string reason;
};

struct Factor {
  std::vector<int32_t> scope;   // variable ids (1-based), size 1 or 2
  std::vector<int32_t> dom;     // domain sizes in scope order
  std::vector<long long> table;
};

struct Model {
  std::vector<Variable> vars;  // 1-based ids in TD/GR, 0-based vector index
  std::vector<Factor> factors;
  std::vector<std::string> op_name_by_x_var;              // 1-based var id -> operator name (only for X vars)
  std::vector<std::vector<std::string>> x_domain_labels;  // 1-based var id -> partition labels
};

struct SolveResult {
  long long objective = 0;
  std::vector<int32_t> var_value;  // 1-based var id -> chosen value in domain
  bool timed_out = false;
  long long solve_runtime_ms = 0;
  size_t dp_peak_table_states = 0;
  size_t dp_peak_live_states = 0;
  long long dp_rss_start_kb = -1;
  long long dp_rss_end_kb = -1;
  long long dp_rss_peak_kb = -1;
};

long long current_rss_kb() {
  std::ifstream in("/proc/self/status");
  if (!in) {
    return -1;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) != 0) {
      continue;
    }
    std::stringstream ss(line.substr(6));
    long long kb = -1;
    ss >> kb;
    if (ss) {
      return kb;
    }
    return -1;
  }
  return -1;
}

std::string trim(const std::string &s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
    ++b;
  }
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
    --e;
  }
  return s.substr(b, e - b);
}

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) {
    parts.push_back(item);
  }
  return parts;
}

long long parse_int_strict(std::string s, const std::string &ctx) {
  s.erase(std::remove(s.begin(), s.end(), ','), s.end());
  s = trim(s);
  if (s.empty()) {
    throw std::runtime_error("Empty integer in " + ctx);
  }
  size_t idx = 0;
  long long val = 0;
  try {
    val = std::stoll(s, &idx);
  } catch (const std::exception &) {
    throw std::runtime_error("Invalid integer '" + s + "' in " + ctx);
  }
  if (idx != s.size()) {
    throw std::runtime_error("Trailing characters in integer '" + s + "' in " + ctx);
  }
  return val;
}

std::string make_choice_key(const std::string &op, const std::string &partition) {
  return op + "\n" + partition;
}

std::pair<std::string, std::string> split_choice_key(const std::string &choice_key) {
  const size_t pos = choice_key.find('\n');
  if (pos == std::string::npos) {
    throw std::runtime_error("Malformed choice key");
  }
  return {choice_key.substr(0, pos), choice_key.substr(pos + 1)};
}

ParsedInstance parse_dump_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open input file: " + path);
  }

  std::unordered_map<std::string, std::vector<std::string>> sections;
  std::string current_section;
  std::string line;

  while (std::getline(in, line)) {
    if (line.rfind("# ", 0) == 0) {
      current_section = trim(line.substr(2));
      sections[current_section];
      continue;
    }
    if (current_section.empty()) {
      continue;
    }
    if (trim(line).empty()) {
      continue;
    }
    sections[current_section].push_back(line);
  }

  ParsedInstance inst;
  std::unordered_set<std::string> known_choice_keys;

  for (const auto &row : sections["operator_costs"]) {
    const auto cols = split(row, ';');
    if (cols.size() != 3) {
      throw std::runtime_error("Bad operator_costs row: " + row);
    }
    const std::string op = trim(cols[0]);
    const std::string part = trim(cols[1]);
    const long long cost = parse_int_strict(cols[2], "operator_costs");

    inst.St[op].push_back(OperatorChoice{part, cost});
    known_choice_keys.insert(make_choice_key(op, part));
  }

  for (const auto &row : sections["tensor_layouts"]) {
    const auto cols = split(row, ';');
    if (cols.size() < 5) {
      throw std::runtime_error("Bad tensor_layouts row: " + row);
    }
    const std::string op = trim(cols[0]);
    const std::string part = trim(cols[1]);
    const std::string tensor = trim(cols[2]);
    const std::string layout = trim(cols[3]);
    const std::string role = trim(cols[4]);

    const std::string ck = make_choice_key(op, part);
    if (known_choice_keys.find(ck) == known_choice_keys.end()) {
      throw std::runtime_error("tensor_layouts references unknown operator choice: " + row);
    }

    bool is_prod = false;
    if (role == "P") {
      is_prod = true;
    } else if (role == "C") {
      is_prod = false;
    } else {
      throw std::runtime_error("Unknown role in tensor_layouts row: " + row);
    }

    inst.choice_uses[ck].push_back(TensorLayoutUse{tensor, layout, is_prod});
  }

  for (const auto &row : sections["tensor_transpose_costs"]) {
    const auto cols = split(row, ';');
    if (cols.size() != 4) {
      throw std::runtime_error("Bad tensor_transpose_costs row: " + row);
    }
    const std::string tensor = trim(cols[0]);
    const std::string src = trim(cols[1]);
    const std::string dst = trim(cols[2]);
    const long long cost = parse_int_strict(cols[3], "tensor_transpose_costs");

    inst.transpose_costs.push_back(TransposeCost{tensor, src, dst, cost});
  }

  for (const auto &row : sections["edges"]) {
    const auto cols = split(row, ';');
    if (cols.size() < 2) {
      throw std::runtime_error("Bad edges row: " + row);
    }
    inst.edges.push_back({trim(cols[0]), trim(cols[1])});
  }

  return inst;
}

void validate_instance(const ParsedInstance &inst) {
  if (inst.St.empty()) {
    throw std::runtime_error("No operators found in # operator_costs");
  }
  for (const auto &kv : inst.St) {
    if (kv.second.empty()) {
      throw std::runtime_error("Operator has empty state space: " + kv.first);
    }
  }

  for (const auto &e : inst.edges) {
    if (inst.St.find(e.first) == inst.St.end()) {
      throw std::runtime_error("Edge source not in operators: " + e.first);
    }
    if (inst.St.find(e.second) == inst.St.end()) {
      throw std::runtime_error("Edge target not in operators: " + e.second);
    }
  }

  for (const auto &kv : inst.choice_uses) {
    std::unordered_map<std::string, std::string> tensor_to_layout;
    for (const auto &u : kv.second) {
      if (!u.is_producer) {
        continue;
      }
      auto it = tensor_to_layout.find(u.tensor);
      if (it == tensor_to_layout.end()) {
        tensor_to_layout.emplace(u.tensor, u.layout);
        continue;
      }
      if (it->second != u.layout) {
        throw std::runtime_error(
            "Choice has conflicting producer layouts for tensor '" + u.tensor + "' in key: " + kv.first);
      }
    }
  }
}

std::map<std::string, std::set<std::string>> collect_tensor_layouts(const ParsedInstance &inst) {
  std::map<std::string, std::set<std::string>> tensor_layouts;
  for (const auto &kv : inst.choice_uses) {
    for (const auto &u : kv.second) {
      tensor_layouts[u.tensor].insert(u.layout);
    }
  }
  for (const auto &tc : inst.transpose_costs) {
    tensor_layouts[tc.tensor].insert(tc.src_layout);
    tensor_layouts[tc.tensor].insert(tc.dst_layout);
  }
  return tensor_layouts;
}

void add_undirected_edge(std::set<std::pair<int, int>> &E, int u, int v) {
  if (u == v) {
    return;
  }
  if (u > v) {
    std::swap(u, v);
  }
  E.insert({u, v});
}

InteractionGraph build_interaction_graph(const ParsedInstance &inst) {
  InteractionGraph G;
  const auto tensor_layouts = collect_tensor_layouts(inst);

  std::vector<std::string> ops;
  for (const auto &kv : inst.St) {
    ops.push_back(kv.first);
  }
  std::sort(ops.begin(), ops.end());

  std::vector<std::string> tensors;
  for (const auto &kv : tensor_layouts) {
    tensors.push_back(kv.first);
  }

  std::unordered_map<std::string, int> x_var;
  std::unordered_map<std::string, int> p_var;
  std::unordered_map<std::string, int> a_var;

  for (const auto &op : ops) {
    const int id = static_cast<int>(G.vars.size()) + 1;
    G.vars.push_back(Variable{VarType::X, op, "", "", static_cast<int>(inst.St.at(op).size())});
    x_var[op] = id;
  }

  for (const auto &t : tensors) {
    const int id = static_cast<int>(G.vars.size()) + 1;
    G.vars.push_back(Variable{VarType::P, "", t, "", static_cast<int>(tensor_layouts.at(t).size())});
    p_var[t] = id;
  }

  for (const auto &t : tensors) {
    for (const auto &l : tensor_layouts.at(t)) {
      const int id = static_cast<int>(G.vars.size()) + 1;
      G.vars.push_back(Variable{VarType::A, "", t, l, 2});
      a_var[t + "\n" + l] = id;
    }
  }

  for (const auto &kv : inst.choice_uses) {
    const auto [op, _part] = split_choice_key(kv.first);
    const int xo = x_var.at(op);
    for (const auto &u : kv.second) {
      if (u.is_producer) {
        add_undirected_edge(G.E, xo, p_var.at(u.tensor));
      } else {
        add_undirected_edge(G.E, xo, a_var.at(u.tensor + "\n" + u.layout));
      }
    }
  }

  for (const auto &tc : inst.transpose_costs) {
    const int pt = p_var.at(tc.tensor);
    const int a = a_var.at(tc.tensor + "\n" + tc.dst_layout);
    add_undirected_edge(G.E, pt, a);
  }

  return G;
}

void write_gr(const InteractionGraph &G, const std::string &path) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot write .gr file: " + path);
  }
  out << "p tw " << G.vars.size() << " " << G.E.size() << "\n";
  for (const auto &e : G.E) {
    out << e.first << " " << e.second << "\n";
  }
}

void write_map(const InteractionGraph &G, const std::string &path) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot write map file: " + path);
  }
  out << "# id;type;name;domain_size\n";
  for (size_t i = 0; i < G.vars.size(); ++i) {
    const int id = static_cast<int>(i) + 1;
    const Variable &v = G.vars[i];
    if (v.type == VarType::X) {
      out << id << ";X;" << v.op << ";" << v.domain_size << "\n";
    } else if (v.type == VarType::P) {
      out << id << ";P;" << v.tensor << ";" << v.domain_size << "\n";
    } else {
      out << id << ";A;" << v.tensor << "|" << v.layout << ";" << v.domain_size << "\n";
    }
  }
}

TD parse_td_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open .td file: " + path);
  }

  TD td;
  bool saw_s = false;
  std::string line;

  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == 'c') {
      continue;
    }

    std::stringstream ss(line);
    std::string head;
    ss >> head;

    if (head == "s") {
      std::string kind;
      ss >> kind;
      if (kind != "td") {
        throw std::runtime_error("Expected 's td ...' header in .td");
      }
      ss >> td.n_bags_declared >> td.width_plus_one_declared >> td.n_graph_vertices;
      if (!ss) {
        throw std::runtime_error("Malformed s td header");
      }
      saw_s = true;
    } else if (head == "b") {
      int id = 0;
      ss >> id;
      if (!ss) {
        throw std::runtime_error("Malformed bag line: " + line);
      }
      if (td.bags.find(id) != td.bags.end()) {
        throw std::runtime_error("Duplicate bag id in .td: " + std::to_string(id));
      }
      std::set<int> bag;
      int v = 0;
      while (ss >> v) {
        bag.insert(v);
      }
      td.bags[id] = std::move(bag);
    } else {
      // Edge line: i j
      int u = 0;
      int v = 0;
      try {
        u = std::stoi(head);
      } catch (const std::exception &) {
        throw std::runtime_error("Malformed .td line: " + line);
      }
      ss >> v;
      if (!ss) {
        throw std::runtime_error("Malformed .td edge line: " + line);
      }
      td.edges.push_back({u, v});
    }
  }

  if (!saw_s) {
    throw std::runtime_error("Missing 's td' header in .td");
  }

  return td;
}

void validate_td_structure(const TD &td) {
  if (td.n_bags_declared != static_cast<int>(td.bags.size())) {
    throw std::runtime_error(".td declared bag count does not match parsed bag lines");
  }

  if (td.bags.empty()) {
    if (!td.edges.empty()) {
      throw std::runtime_error(".td has edges but no bags");
    }
    return;
  }

  std::unordered_map<int, std::vector<int>> adj;
  for (const auto &kv : td.bags) {
    adj[kv.first] = {};
  }
  for (const auto &e : td.edges) {
    if (td.bags.find(e.first) == td.bags.end() || td.bags.find(e.second) == td.bags.end()) {
      throw std::runtime_error(".td edge references unknown bag id");
    }
    if (e.first == e.second) {
      throw std::runtime_error(".td contains self-loop edge");
    }
    adj[e.first].push_back(e.second);
    adj[e.second].push_back(e.first);
  }

  if (td.edges.size() != td.bags.size() - 1) {
    throw std::runtime_error(".td does not have |E|=|V|-1 (not a tree)");
  }

  std::unordered_set<int> vis;
  std::vector<int> st;
  st.push_back(td.bags.begin()->first);
  while (!st.empty()) {
    const int u = st.back();
    st.pop_back();
    if (!vis.insert(u).second) {
      continue;
    }
    for (int v : adj[u]) {
      if (vis.find(v) == vis.end()) {
        st.push_back(v);
      }
    }
  }

  if (vis.size() != td.bags.size()) {
    throw std::runtime_error(".td is disconnected");
  }
}

int choose_root_bag_id(const TD &td) {
  for (const auto &kv : td.bags) {
    if (kv.second.empty()) {
      return kv.first;
    }
  }
  return td.bags.begin()->first;
}

RootedTD root_td(const TD &td, int root_bag_id) {
  if (td.bags.empty()) {
    return RootedTD{};
  }

  std::vector<int> ids;
  ids.reserve(td.bags.size());
  for (const auto &kv : td.bags) {
    ids.push_back(kv.first);
  }
  std::sort(ids.begin(), ids.end());

  if (std::find(ids.begin(), ids.end(), root_bag_id) == ids.end()) {
    throw std::runtime_error("Requested root bag id does not exist");
  }

  std::unordered_map<int, int> old_to_new;
  std::vector<int> new_to_old(ids.size() + 1, -1);
  for (size_t i = 0; i < ids.size(); ++i) {
    old_to_new[ids[i]] = static_cast<int>(i) + 1;
    new_to_old[static_cast<int>(i) + 1] = ids[i];
  }

  const int n = static_cast<int>(ids.size());
  std::vector<std::vector<int>> adj(n + 1);
  for (const auto &e : td.edges) {
    const int u = old_to_new.at(e.first);
    const int v = old_to_new.at(e.second);
    adj[u].push_back(v);
    adj[v].push_back(u);
  }

  RootedTD r;
  r.bag.resize(n + 1);
  r.ch.resize(n + 1);

  for (int i = 1; i <= n; ++i) {
    r.bag[i] = td.bags.at(new_to_old[i]);
  }

  r.root = old_to_new.at(root_bag_id);
  std::vector<int> parent(n + 1, 0);
  std::vector<int> st = {r.root};
  parent[r.root] = -1;

  while (!st.empty()) {
    const int u = st.back();
    st.pop_back();
    for (int v : adj[u]) {
      if (v == parent[u]) {
        continue;
      }
      if (parent[v] != 0) {
        continue;
      }
      parent[v] = u;
      r.ch[u].push_back(v);
      st.push_back(v);
    }
  }

  for (int i = 1; i <= n; ++i) {
    if (i == r.root) {
      continue;
    }
    if (parent[i] == 0) {
      throw std::runtime_error("Failed to orient TD as rooted tree");
    }
  }

  return r;
}

int add_node(RootedTD &t, const std::set<int> &bag) {
  t.bag.push_back(bag);
  t.ch.push_back({});
  return static_cast<int>(t.bag.size()) - 1;
}

void binarize_children(RootedTD &t, int u) {
  for (int v : t.ch[u]) {
    binarize_children(t, v);
  }
  auto children = t.ch[u];
  if (children.size() <= 2) {
    return;
  }

  t.ch[u].clear();
  int cur = u;
  size_t i = 0;
  while (children.size() - i > 2) {
    const int join = add_node(t, t.bag[u]);
    t.ch[cur].push_back(children[i]);
    t.ch[cur].push_back(join);
    cur = join;
    ++i;
  }
  t.ch[cur].push_back(children[i]);
  t.ch[cur].push_back(children[i + 1]);
}

void binarize_rooted(RootedTD &t) {
  if (t.root == 0) {
    return;
  }
  binarize_children(t, t.root);
}

std::vector<int> set_diff_sorted(const std::set<int> &a, const std::set<int> &b) {
  std::vector<int> out;
  for (int x : a) {
    if (b.find(x) == b.end()) {
      out.push_back(x);
    }
  }
  return out;
}

void expand_edge_differences_inplace_dfs(RootedTD &t, int u) {
  for (size_t i = 0; i < t.ch[u].size(); ++i) {
    const int v = t.ch[u][i];
    expand_edge_differences_inplace_dfs(t, v);

    std::set<int> cur_bag = t.bag[u];
    int first = v;
    int cur = -1;

    const auto to_forget = set_diff_sorted(cur_bag, t.bag[v]);
    for (int x : to_forget) {
      cur_bag.erase(x);
      const int n = add_node(t, cur_bag);
      if (first == v) {
        first = n;
      } else {
        t.ch[cur].push_back(n);
      }
      cur = n;
    }

    const auto to_intro = set_diff_sorted(t.bag[v], cur_bag);
    for (int x : to_intro) {
      cur_bag.insert(x);
      const int n = add_node(t, cur_bag);
      if (first == v) {
        first = n;
      } else {
        t.ch[cur].push_back(n);
      }
      cur = n;
    }

    if (first != v) {
      t.ch[cur].push_back(v);
      t.ch[u][i] = first;
    }
  }
}

void expand_edge_differences_inplace(RootedTD &t) {
  if (t.root == 0) {
    return;
  }
  expand_edge_differences_inplace_dfs(t, t.root);
}

void compress_unary_equal_dfs(RootedTD &t, int u) {
  while (t.ch[u].size() == 1) {
    const int c = t.ch[u][0];
    if (t.bag[c] != t.bag[u]) {
      break;
    }
    t.ch[u] = t.ch[c];
  }
  for (int v : t.ch[u]) {
    compress_unary_equal_dfs(t, v);
  }
}

void compress_unary_equal(RootedTD &t) {
  if (t.root == 0) {
    return;
  }
  compress_unary_equal_dfs(t, t.root);
}

void enforce_join_children_equal_dfs(RootedTD &t, int u) {
  for (int v : t.ch[u]) {
    enforce_join_children_equal_dfs(t, v);
  }
  if (t.ch[u].size() != 2) {
    return;
  }
  for (size_t i = 0; i < 2; ++i) {
    const int v = t.ch[u][i];
    if (t.bag[v] == t.bag[u]) {
      continue;
    }
    const int shim = add_node(t, t.bag[u]);
    t.ch[shim].push_back(v);
    t.ch[u][i] = shim;
  }
}

void enforce_join_children_equal(RootedTD &t) {
  if (t.root == 0) {
    return;
  }
  enforce_join_children_equal_dfs(t, t.root);
}

void ensure_empty_root(RootedTD &t) {
  if (t.root == 0) {
    return;
  }
  if (t.bag[t.root].empty()) {
    return;
  }

  const int old_root = t.root;
  int cur = add_node(t, {});
  t.root = cur;

  std::set<int> cur_bag;
  for (int x : t.bag[old_root]) {
    cur_bag.insert(x);
    const int n = add_node(t, cur_bag);
    t.ch[cur].push_back(n);
    cur = n;
  }
  t.ch[cur].push_back(old_root);
}

void ensure_empty_leaves(RootedTD &t) {
  if (t.root == 0) {
    return;
  }

  std::vector<int> leaves;
  for (int i = 1; i < static_cast<int>(t.bag.size()); ++i) {
    if (t.ch[i].empty()) {
      leaves.push_back(i);
    }
  }

  for (int leaf : leaves) {
    if (t.bag[leaf].empty()) {
      continue;
    }
    int cur = leaf;
    std::vector<int> elems(t.bag[leaf].begin(), t.bag[leaf].end());
    for (int x : elems) {
      auto nb = t.bag[cur];
      nb.erase(x);
      const int n = add_node(t, nb);
      t.ch[cur].push_back(n);
      cur = n;
    }
  }
}

NiceCheckResult check_nice(const RootedTD &t) {
  if (t.root == 0) {
    return NiceCheckResult{false, -1, "empty rooted TD"};
  }

  std::vector<char> reach(t.bag.size(), 0);
  std::vector<int> st = {t.root};
  while (!st.empty()) {
    const int u = st.back();
    st.pop_back();
    if (reach[u]) {
      continue;
    }
    reach[u] = 1;
    for (int v : t.ch[u]) {
      st.push_back(v);
    }
  }

  if (!t.bag[t.root].empty()) {
    return NiceCheckResult{false, t.root, "root bag is not empty"};
  }

  for (int u = 1; u < static_cast<int>(t.bag.size()); ++u) {
    if (!reach[u]) {
      continue;
    }
    const size_t k = t.ch[u].size();

    if (k == 0) {
      if (!t.bag[u].empty()) {
        return NiceCheckResult{false, u, "leaf bag is not empty"};
      }
      continue;
    }

    if (k == 1) {
      const int c = t.ch[u][0];
      size_t diff = 0;
      for (int x : t.bag[u]) {
        if (t.bag[c].find(x) == t.bag[c].end()) {
          ++diff;
        }
      }
      for (int x : t.bag[c]) {
        if (t.bag[u].find(x) == t.bag[u].end()) {
          ++diff;
        }
      }
      if (diff != 1) {
        return NiceCheckResult{false, u, "single-child node is not introduce/forget (bag diff != 1)"};
      }
      continue;
    }

    if (k == 2) {
      const int c1 = t.ch[u][0];
      const int c2 = t.ch[u][1];
      if (t.bag[u] != t.bag[c1] || t.bag[u] != t.bag[c2]) {
        return NiceCheckResult{false, u, "two-child node is not join (bags not equal)"};
      }
      continue;
    }

    return NiceCheckResult{false, u, "node has more than two children"};
  }

  return NiceCheckResult{true, -1, "ok"};
}

RootedTD convert_to_nice(const RootedTD &src) {
  RootedTD t = src;
  binarize_rooted(t);
  expand_edge_differences_inplace(t);
  compress_unary_equal(t);
  enforce_join_children_equal(t);
  ensure_empty_root(t);
  ensure_empty_leaves(t);
  compress_unary_equal(t);
  return t;
}

TD rooted_to_td(const RootedTD &r, int n_graph_vertices) {
  TD td;
  td.n_graph_vertices = n_graph_vertices;

  std::vector<char> reach(r.bag.size(), 0);
  std::vector<int> st = {r.root};
  while (!st.empty()) {
    const int u = st.back();
    st.pop_back();
    if (reach[u]) {
      continue;
    }
    reach[u] = 1;
    for (int v : r.ch[u]) {
      st.push_back(v);
    }
  }

  std::vector<int> nodes;
  std::vector<int> stack = {r.root};
  while (!stack.empty()) {
    const int u = stack.back();
    stack.pop_back();
    nodes.push_back(u);
    // Reverse push to keep child order stable in preorder numbering.
    for (auto it = r.ch[u].rbegin(); it != r.ch[u].rend(); ++it) {
      stack.push_back(*it);
    }
  }

  std::unordered_map<int, int> ren;
  for (size_t i = 0; i < nodes.size(); ++i) {
    ren[nodes[i]] = static_cast<int>(i) + 1;
  }

  td.n_bags_declared = static_cast<int>(nodes.size());

  size_t max_bag_size = 0;
  for (int old_u : nodes) {
    const int u = ren.at(old_u);
    td.bags[u] = r.bag[old_u];
    max_bag_size = std::max(max_bag_size, r.bag[old_u].size());
    for (int old_v : r.ch[old_u]) {
      if (!reach[old_v]) {
        continue;
      }
      td.edges.push_back({u, ren.at(old_v)});
    }
  }

  td.width_plus_one_declared = static_cast<int>(max_bag_size);
  return td;
}

void write_rooted_td(const RootedTD &r, int n_graph_vertices, const std::string &path,
                     const std::string &label_for_summary) {
  if (r.root == 0) {
    throw std::runtime_error("Cannot write rooted TD: empty root");
  }

  std::vector<char> reach(r.bag.size(), 0);
  std::vector<int> st = {r.root};
  while (!st.empty()) {
    const int u = st.back();
    st.pop_back();
    if (reach[u]) {
      continue;
    }
    reach[u] = 1;
    for (int v : r.ch[u]) {
      st.push_back(v);
    }
  }

  std::vector<int> nodes;
  nodes.reserve(r.bag.size() - 1);
  std::vector<int> stack = {r.root};
  while (!stack.empty()) {
    const int u = stack.back();
    stack.pop_back();
    nodes.push_back(u);
    for (auto it = r.ch[u].rbegin(); it != r.ch[u].rend(); ++it) {
      stack.push_back(*it);
    }
  }

  std::unordered_map<int, int> ren;
  ren.reserve(nodes.size() * 2 + 1);
  for (size_t i = 0; i < nodes.size(); ++i) {
    ren[nodes[i]] = static_cast<int>(i) + 1;
  }

  int edge_count = 0;
  size_t max_bag_size = 0;
  for (int old_u : nodes) {
    max_bag_size = std::max(max_bag_size, r.bag[old_u].size());
    for (int old_v : r.ch[old_u]) {
      if (reach[old_v]) {
        ++edge_count;
      }
    }
  }

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot write .td file: " + path);
  }

  out << "s td " << nodes.size() << " " << static_cast<int>(max_bag_size) << " " << n_graph_vertices << "\n";
  for (int old_u : nodes) {
    out << "b " << ren.at(old_u);
    for (int v : r.bag[old_u]) {
      out << " " << v;
    }
    out << "\n";
  }
  for (int old_u : nodes) {
    const int u = ren.at(old_u);
    for (int old_v : r.ch[old_u]) {
      if (!reach[old_v]) {
        continue;
      }
      out << u << " " << ren.at(old_v) << "\n";
    }
  }

  std::cout << label_for_summary << ": bags=" << nodes.size() << ", edges=" << edge_count
            << ", width+1=" << max_bag_size << ", n_graph=" << n_graph_vertices << "\n";
}

void write_td(const TD &td, const std::string &path) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot write .td file: " + path);
  }

  out << "s td " << td.n_bags_declared << " " << td.width_plus_one_declared << " " << td.n_graph_vertices << "\n";
  for (const auto &kv : td.bags) {
    out << "b " << kv.first;
    for (int v : kv.second) {
      out << " " << v;
    }
    out << "\n";
  }
  for (const auto &e : td.edges) {
    out << e.first << " " << e.second << "\n";
  }
}

void print_instance_summary(const ParsedInstance &inst, const std::string &input_path,
                            const InteractionGraph *G = nullptr) {
  size_t total_states = 0;
  for (const auto &kv : inst.St) {
    total_states += kv.second.size();
  }

  std::unordered_set<std::string> tensors;
  size_t total_tensor_uses = 0;
  size_t producer_uses = 0;
  size_t consumer_uses = 0;
  for (const auto &kv : inst.choice_uses) {
    for (const auto &u : kv.second) {
      ++total_tensor_uses;
      tensors.insert(u.tensor);
      if (u.is_producer) {
        ++producer_uses;
      } else {
        ++consumer_uses;
      }
    }
  }

  std::cout << "input: " << input_path << "\n";
  std::cout << "operators: " << inst.St.size() << "\n";
  std::cout << "choices (sum |St(o)|): " << total_states << "\n";
  std::cout << "edges: " << inst.edges.size() << "\n";
  std::cout << "tensors: " << tensors.size() << "\n";
  std::cout << "tensor layout uses: " << total_tensor_uses
            << " (P=" << producer_uses << ", C=" << consumer_uses << ")\n";
  std::cout << "transpose entries: " << inst.transpose_costs.size() << "\n";
  if (G != nullptr) {
    std::cout << "interaction graph: |U|=" << G->vars.size() << ", |F|=" << G->E.size() << "\n";
  }
}

void print_td_summary(const TD &td, const std::string &label) {
  std::cout << label << ": bags=" << td.bags.size() << ", edges=" << td.edges.size()
            << ", width+1=" << td.width_plus_one_declared << ", n_graph=" << td.n_graph_vertices << "\n";
}

void write_dump_with_layout_selection(const std::string &input_path, const std::string &out_path,
                                      const std::vector<std::pair<std::string, std::string>> &layout_sel) {
  std::ifstream in(input_path);
  if (!in) {
    throw std::runtime_error("Cannot open input dump for writing output: " + input_path);
  }
  std::ofstream out(out_path);
  if (!out) {
    throw std::runtime_error("Cannot open output path: " + out_path);
  }

  std::string line;
  bool skip_layout_section = false;
  while (std::getline(in, line)) {
    if (line.rfind("# ", 0) == 0) {
      const std::string sec = trim(line.substr(2));
      if (sec == "layout_selection") {
        skip_layout_section = true;
        continue;
      }
      skip_layout_section = false;
    }
    if (!skip_layout_section) {
      out << line << "\n";
    }
  }

  out << "\n# layout_selection\n";
  for (const auto &p : layout_sel) {
    out << p.first << ";" << p.second << "\n";
  }
}

long long sat_add(long long a, long long b) {
  static constexpr long long INF = std::numeric_limits<long long>::max() / 4;
  if (a >= INF || b >= INF) {
    return INF;
  }
  if (a > INF - b) {
    return INF;
  }
  return a + b;
}

Model build_model(const ParsedInstance &inst) {
  static constexpr long long INF = std::numeric_limits<long long>::max() / 4;

  Model model;
  const auto tensor_layouts = collect_tensor_layouts(inst);

  std::vector<std::string> ops;
  for (const auto &kv : inst.St) {
    ops.push_back(kv.first);
  }
  std::sort(ops.begin(), ops.end());

  std::vector<std::string> tensors;
  for (const auto &kv : tensor_layouts) {
    tensors.push_back(kv.first);
  }

  // Variable ids match emitted .gr ids exactly.
  std::unordered_map<std::string, int> x_var;
  std::unordered_map<std::string, int> p_var;
  std::unordered_map<std::string, int> a_var;

  std::unordered_map<std::string, std::vector<std::string>> op_choices_sorted;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> x_choice_index;
  for (const auto &op : ops) {
    std::vector<std::string> parts;
    parts.reserve(inst.St.at(op).size());
    for (const auto &ch : inst.St.at(op)) {
      parts.push_back(ch.partition);
    }
    std::sort(parts.begin(), parts.end());
    parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
    op_choices_sorted[op] = parts;
    for (size_t i = 0; i < parts.size(); ++i) {
      x_choice_index[op][parts[i]] = static_cast<int>(i);
    }
  }

  std::unordered_map<std::string, std::vector<std::string>> p_layouts;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> p_layout_index;
  for (const auto &t : tensors) {
    std::vector<std::string> ls(tensor_layouts.at(t).begin(), tensor_layouts.at(t).end());
    p_layouts[t] = ls;
    for (size_t i = 0; i < ls.size(); ++i) {
      p_layout_index[t][ls[i]] = static_cast<int>(i);
    }
  }

  // X_o variables.
  for (const auto &op : ops) {
    const int id = static_cast<int>(model.vars.size()) + 1;
    model.vars.push_back(Variable{VarType::X, op, "", "", static_cast<int>(op_choices_sorted[op].size())});
    x_var[op] = id;
  }

  // P_t variables.
  for (const auto &t : tensors) {
    const int id = static_cast<int>(model.vars.size()) + 1;
    model.vars.push_back(Variable{VarType::P, "", t, "", static_cast<int>(p_layouts[t].size())});
    p_var[t] = id;
  }

  // A_{t,l} variables.
  for (const auto &t : tensors) {
    for (const auto &l : p_layouts[t]) {
      const int id = static_cast<int>(model.vars.size()) + 1;
      model.vars.push_back(Variable{VarType::A, "", t, l, 2});
      a_var[t + "\n" + l] = id;
    }
  }

  // Unary opcost factors on X_o.
  model.op_name_by_x_var.assign(model.vars.size() + 1, "");
  model.x_domain_labels.assign(model.vars.size() + 1, {});
  for (const auto &op : ops) {
    const int id = x_var.at(op);
    model.op_name_by_x_var[id] = op;
    model.x_domain_labels[id] = op_choices_sorted.at(op);
  }

  for (const auto &op : ops) {
    std::vector<long long> table(op_choices_sorted[op].size(), 0);
    std::unordered_map<std::string, long long> part_cost;
    for (const auto &ch : inst.St.at(op)) {
      auto it = part_cost.find(ch.partition);
      if (it == part_cost.end()) {
        part_cost[ch.partition] = ch.opcost;
      } else {
        it->second = std::min(it->second, ch.opcost);
      }
    }
    for (size_t i = 0; i < op_choices_sorted[op].size(); ++i) {
      table[i] = part_cost.at(op_choices_sorted[op][i]);
    }
    model.factors.push_back(Factor{{x_var.at(op)}, {static_cast<int>(table.size())}, table});
  }

  // Collect X->P and X->A hard requirements.
  std::map<std::pair<std::string, std::string>, std::map<int, int>> req_xp;
  std::map<std::tuple<std::string, std::string, std::string>, std::set<int>> req_xa;

  for (const auto &kv : inst.choice_uses) {
    const auto [op, part] = split_choice_key(kv.first);
    const int x_idx = x_choice_index.at(op).at(part);

    for (const auto &u : kv.second) {
      if (u.is_producer) {
        const int p_idx = p_layout_index.at(u.tensor).at(u.layout);
        auto &mp = req_xp[{op, u.tensor}];
        auto it = mp.find(x_idx);
        if (it == mp.end()) {
          mp[x_idx] = p_idx;
        } else if (it->second != p_idx) {
          throw std::runtime_error("Conflicting producer requirement in model build");
        } else {
          // duplicate same requirement, ignore
        }
      } else {
        req_xa[{op, u.tensor, u.layout}].insert(x_idx);
      }
    }
  }

  for (const auto &kv : req_xp) {
    const std::string &op = kv.first.first;
    const std::string &t = kv.first.second;
    const int dx = model.vars[x_var.at(op) - 1].domain_size;
    const int dp = model.vars[p_var.at(t) - 1].domain_size;
    std::vector<long long> table(dx * dp, 0);
    for (const auto &pair : kv.second) {
      const int x_idx = pair.first;
      const int p_req = pair.second;
      for (int p = 0; p < dp; ++p) {
        if (p != p_req) {
          table[x_idx + dx * p] = INF;
        }
      }
    }
    model.factors.push_back(Factor{{x_var.at(op), p_var.at(t)}, {dx, dp}, table});
  }

  for (const auto &kv : req_xa) {
    const std::string &op = std::get<0>(kv.first);
    const std::string &t = std::get<1>(kv.first);
    const std::string &l = std::get<2>(kv.first);
    const int dx = model.vars[x_var.at(op) - 1].domain_size;
    std::vector<long long> table(dx * 2, 0);
    for (int x_idx : kv.second) {
      table[x_idx + dx * 0] = INF;
    }
    model.factors.push_back(Factor{{x_var.at(op), a_var.at(t + "\n" + l)}, {dx, 2}, table});
  }

  // Transpose factors on (P_t, A_{t,l}).
  std::unordered_map<std::string, long long> tcost_lookup;
  std::map<std::string, std::set<std::string>> dst_layouts_with_tcost;
  for (const auto &tc : inst.transpose_costs) {
    tcost_lookup[tc.tensor + "\n" + tc.src_layout + "\n" + tc.dst_layout] = tc.tcost;
    dst_layouts_with_tcost[tc.tensor].insert(tc.dst_layout);
  }

  for (const auto &t : tensors) {
    const int p_id = p_var.at(t);
    const int dp = model.vars[p_id - 1].domain_size;
    const auto it_dst = dst_layouts_with_tcost.find(t);
    if (it_dst == dst_layouts_with_tcost.end()) {
      continue;
    }
    for (const auto &l_dst : it_dst->second) {
      const int a_id = a_var.at(t + "\n" + l_dst);
      std::vector<long long> table(dp * 2, 0);
      for (int p = 0; p < dp; ++p) {
        const std::string &l_src = p_layouts[t][p];
        const std::string key = t + "\n" + l_src + "\n" + l_dst;
        const auto it = tcost_lookup.find(key);
        const long long c = (it == tcost_lookup.end() ? 0LL : it->second);
        table[p + dp * 1] = c;
      }
      model.factors.push_back(Factor{{p_id, a_id}, {dp, 2}, table});
    }
  }

  return model;
}

size_t state_count_from_domains(const std::vector<int32_t> &dom) {
  size_t n = 1;
  for (int32_t d : dom) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

void decode_state(size_t idx, const std::vector<int32_t> &dom, std::vector<int32_t> &out) {
  out.assign(dom.size(), 0);
  for (size_t i = 0; i < dom.size(); ++i) {
    out[i] = static_cast<int32_t>(idx % static_cast<size_t>(dom[i]));
    idx /= static_cast<size_t>(dom[i]);
  }
}

size_t encode_state(const std::vector<int32_t> &vals, const std::vector<int32_t> &dom) {
  size_t idx = 0;
  size_t mul = 1;
  for (size_t i = 0; i < dom.size(); ++i) {
    idx += mul * static_cast<size_t>(vals[i]);
    mul *= static_cast<size_t>(dom[i]);
  }
  return idx;
}

long long eval_factor(const Factor &f, const std::vector<int32_t> &vals_by_scope) {
  if (f.scope.size() == 1) {
    return f.table.at(static_cast<size_t>(vals_by_scope[0]));
  }
  const size_t idx = static_cast<size_t>(vals_by_scope[0]) +
                     static_cast<size_t>(f.dom[0]) * static_cast<size_t>(vals_by_scope[1]);
  return f.table.at(idx);
}

SolveResult solve_exact_objective_on_nice_td(
    const Model &model, const RootedTD &t, bool recover_assignment, long long timeout_ms) {
  static constexpr long long INF = std::numeric_limits<long long>::max() / 4;

  const NiceCheckResult chk = check_nice(t);
  if (!chk.ok) {
    throw std::runtime_error("solve requires nice TD; violation at node " + std::to_string(chk.node) + ": " +
                             chk.reason);
  }

  const auto t_start = std::chrono::steady_clock::now();
  const int nvar = static_cast<int>(model.vars.size());
  const int nnode = static_cast<int>(t.bag.size()) - 1;
  if (nnode <= 0) {
    SolveResult r;
    r.objective = 0;
    r.var_value.assign(static_cast<size_t>(nvar) + 1, -1);
    return r;
  }

  std::vector<char> reach(t.bag.size(), 0);
  std::vector<int> preorder;
  std::vector<int> depth(t.bag.size(), -1);
  {
    std::vector<int> st = {t.root};
    depth[t.root] = 0;
    while (!st.empty()) {
      const int u = st.back();
      st.pop_back();
      if (reach[u]) {
        continue;
      }
      reach[u] = 1;
      preorder.push_back(u);
      for (auto it = t.ch[u].rbegin(); it != t.ch[u].rend(); ++it) {
        depth[*it] = depth[u] + 1;
        st.push_back(*it);
      }
    }
  }

  // Bag variable lists and position maps.
  std::vector<std::vector<int32_t>> bag_vars(t.bag.size());
  std::vector<std::vector<int32_t>> bag_dom(t.bag.size());
  std::vector<std::vector<int32_t>> pos_in_bag(t.bag.size(),
                                               std::vector<int32_t>(static_cast<size_t>(nvar) + 1, -1));
  for (int u : preorder) {
    for (int v : t.bag[u]) {
      bag_vars[u].push_back(static_cast<int32_t>(v));
    }
    for (int32_t v : bag_vars[u]) {
      bag_dom[u].push_back(static_cast<int32_t>(model.vars[v - 1].domain_size));
      pos_in_bag[u][static_cast<size_t>(v)] = static_cast<int32_t>(bag_dom[u].size()) - 1;
    }
  }

  // Assign each factor to the deepest bag containing its scope.
  std::vector<std::vector<int>> node_factors(t.bag.size());
  for (int fid = 0; fid < static_cast<int>(model.factors.size()); ++fid) {
    const Factor &f = model.factors[fid];
    int best = -1;
    int best_depth = -1;
    for (int u : preorder) {
      bool ok = true;
      for (int32_t x : f.scope) {
        if (pos_in_bag[u][static_cast<size_t>(x)] < 0) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        continue;
      }
      if (depth[u] > best_depth || (depth[u] == best_depth && u < best)) {
        best = u;
        best_depth = depth[u];
      }
    }
    if (best < 0) {
      std::ostringstream oss;
      oss << "Factor scope not covered by TD bag: {";
      for (size_t i = 0; i < f.scope.size(); ++i) {
        if (i) {
          oss << ",";
        }
        oss << f.scope[i];
      }
      oss << "}";
      throw std::runtime_error(oss.str());
    }
    node_factors[best].push_back(fid);
  }

  // Postorder traversal.
  std::vector<int> postorder;
  std::function<void(int)> dfs_post = [&](int u) {
    for (int v : t.ch[u]) {
      dfs_post(v);
    }
    postorder.push_back(u);
  };
  dfs_post(t.root);

  std::vector<std::vector<long long>> dp(t.bag.size());
  size_t live_states = 0;
  size_t peak_table_states = 0;
  size_t peak_states = 0;
  const long long rss_start_kb = current_rss_kb();
  long long rss_peak_kb = rss_start_kb;
  auto sample_rss = [&]() {
    const long long now_kb = current_rss_kb();
    if (now_kb < 0) {
      return;
    }
    if (rss_peak_kb < 0 || now_kb > rss_peak_kb) {
      rss_peak_kb = now_kb;
    }
  };
  auto check_timeout = [&]() {
    if (timeout_ms <= 0) {
      return;
    }
    sample_rss();
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count();
    if (elapsed > timeout_ms) {
      throw std::runtime_error("DP timeout exceeded");
    }
  };
  std::vector<std::vector<uint32_t>> back_one;
  if (recover_assignment) {
    back_one.resize(t.bag.size());
  }
  auto release_dp = [&](int node) {
    live_states -= dp[node].size();
    std::vector<long long>().swap(dp[node]);
  };
  std::vector<int32_t> vals;
  std::vector<int32_t> fvals;

  for (int u : postorder) {
    const auto &vars_u = bag_vars[u];
    const auto &dom_u = bag_dom[u];
    const size_t states_u = state_count_from_domains(dom_u);
    dp[u].assign(states_u, INF);
    peak_table_states = std::max(peak_table_states, states_u);
    live_states += states_u;
    peak_states = std::max(peak_states, live_states);

    auto local_cost = [&](const std::vector<int32_t> &bag_assignment) -> long long {
      long long c = 0;
      for (int fid : node_factors[u]) {
        const Factor &f = model.factors[fid];
        fvals.assign(f.scope.size(), 0);
        for (size_t i = 0; i < f.scope.size(); ++i) {
          const int32_t p = pos_in_bag[u][static_cast<size_t>(f.scope[i])];
          if (p < 0) {
            throw std::runtime_error("Factor variable missing in assigned bag");
          }
          fvals[i] = bag_assignment[p];
        }
        c = sat_add(c, eval_factor(f, fvals));
      }
      return c;
    };

    const size_t k = t.ch[u].size();
    if (k == 0) {
      for (size_t s = 0; s < states_u; ++s) {
        if ((s & 4095ULL) == 0) {
          check_timeout();
        }
        decode_state(s, dom_u, vals);
        dp[u][s] = local_cost(vals);
      }
      continue;
    }

    if (k == 2) {
      const int c1 = t.ch[u][0];
      const int c2 = t.ch[u][1];
      for (size_t s = 0; s < states_u; ++s) {
        if ((s & 4095ULL) == 0) {
          check_timeout();
        }
        decode_state(s, dom_u, vals);
        long long v = sat_add(dp[c1][s], dp[c2][s]);
        v = sat_add(v, local_cost(vals));
        dp[u][s] = v;
      }
      release_dp(c1);
      release_dp(c2);
      continue;
    }

    // One-child node: either introduce or forget.
    const int c = t.ch[u][0];
    const auto &vars_c = bag_vars[c];
    const auto &dom_c = bag_dom[c];

    if (vars_u.size() == vars_c.size() + 1) {
      // Introduce.
      for (size_t s = 0; s < states_u; ++s) {
        if ((s & 4095ULL) == 0) {
          check_timeout();
        }
        decode_state(s, dom_u, vals);
        std::vector<int32_t> child_vals;
        child_vals.reserve(vars_c.size());
        for (int32_t v : vars_c) {
          const int32_t p = pos_in_bag[u][static_cast<size_t>(v)];
          child_vals.push_back(vals[p]);
        }
        const size_t sc = encode_state(child_vals, dom_c);
        long long v = sat_add(dp[c][sc], local_cost(vals));
        dp[u][s] = v;
      }
      release_dp(c);
    } else if (vars_u.size() + 1 == vars_c.size()) {
      // Forget.
      if (recover_assignment) {
        back_one[u].assign(states_u, 0);
      }
      int32_t forgotten = -1;
      for (int32_t v : vars_c) {
        if (pos_in_bag[u][static_cast<size_t>(v)] < 0) {
          forgotten = v;
          break;
        }
      }
      if (forgotten < 0) {
        throw std::runtime_error("Could not identify forgotten variable");
      }
      const int32_t d_forget = static_cast<int32_t>(model.vars[forgotten - 1].domain_size);
      for (size_t s = 0; s < states_u; ++s) {
        if ((s & 4095ULL) == 0) {
          check_timeout();
        }
        decode_state(s, dom_u, vals);
        const long long lc = local_cost(vals);
        long long best = INF;
        size_t best_sc = 0;
        for (int32_t fv = 0; fv < d_forget; ++fv) {
          std::vector<int32_t> child_vals;
          child_vals.reserve(vars_c.size());
          for (int32_t v : vars_c) {
            if (v == forgotten) {
              child_vals.push_back(fv);
            } else {
              const int32_t p = pos_in_bag[u][static_cast<size_t>(v)];
              child_vals.push_back(vals[p]);
            }
          }
          const size_t sc = encode_state(child_vals, dom_c);
          const long long cand = dp[c][sc];
          if (cand < best) {
            best = cand;
            best_sc = sc;
          }
        }
        if (recover_assignment) {
          back_one[u][s] = static_cast<uint32_t>(best_sc);
        }
        dp[u][s] = sat_add(best, lc);
      }
      release_dp(c);
    } else {
      throw std::runtime_error("Unexpected one-child bag sizes in nice TD");
    }
  }

  if (dp[t.root].size() != 1) {
    throw std::runtime_error("Root bag is expected to be empty for nice TD");
  }
  SolveResult res;
  res.objective = dp[t.root][0];
  res.dp_peak_table_states = peak_table_states;
  res.dp_peak_live_states = peak_states;
  res.solve_runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t_start).count();
  sample_rss();
  res.dp_rss_start_kb = rss_start_kb;
  res.dp_rss_peak_kb = rss_peak_kb;
  res.dp_rss_end_kb = current_rss_kb();
  if (!recover_assignment) {
    return res;
  }
  res.var_value.assign(static_cast<size_t>(nvar) + 1, -1);

  // Recover one optimal assignment by following backpointers from root.
  std::function<void(int, size_t)> rec = [&](int u, size_t state_u) {
    std::vector<int32_t> a_u;
    decode_state(state_u, bag_dom[u], a_u);
    for (size_t i = 0; i < bag_vars[u].size(); ++i) {
      const int32_t var = bag_vars[u][i];
      const int32_t val = a_u[i];
      if (res.var_value[var] < 0) {
        res.var_value[var] = val;
      }
    }

    if (t.ch[u].empty()) {
      return;
    }
    if (t.ch[u].size() == 2) {
      rec(t.ch[u][0], state_u);
      rec(t.ch[u][1], state_u);
      return;
    }
    const int c = t.ch[u][0];
    if (bag_vars[u].size() == bag_vars[c].size() + 1) {
      // Introduce: child assignment is parent assignment restricted to child bag vars.
      std::vector<int32_t> child_vals;
      child_vals.reserve(bag_vars[c].size());
      for (int32_t v : bag_vars[c]) {
        const int32_t p = pos_in_bag[u][static_cast<size_t>(v)];
        child_vals.push_back(a_u[p]);
      }
      const size_t state_c = encode_state(child_vals, bag_dom[c]);
      rec(c, state_c);
      return;
    }
    if (back_one[u].empty()) {
      throw std::runtime_error("Missing backpointer for forget node");
    }
    const size_t state_c = static_cast<size_t>(back_one[u][state_u]);
    rec(c, state_c);
  };

  rec(t.root, 0);
  res.solve_runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t_start).count();
  sample_rss();
  res.dp_rss_peak_kb = rss_peak_kb;
  res.dp_rss_end_kb = current_rss_kb();
  return res;
}

int run(int argc, char **argv) {
  std::string input_path;
  std::string out_gr_path;
  std::string out_map_path;

  std::string input_td_path;
  bool check_nice_flag = false;
  std::string out_nice_td_path;
  bool solve_objective_flag = false;
  std::string out_solved_dump_path;
  long long solve_timeout_ms = 600000;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (arg == "--emit-gr" && i + 1 < argc) {
      out_gr_path = argv[++i];
    } else if (arg == "--emit-map" && i + 1 < argc) {
      out_map_path = argv[++i];
    } else if (arg == "--input-td" && i + 1 < argc) {
      input_td_path = argv[++i];
    } else if (arg == "--check-nice") {
      check_nice_flag = true;
    } else if (arg == "--to-nice" && i + 1 < argc) {
      out_nice_td_path = argv[++i];
    } else if (arg == "--solve-objective") {
      solve_objective_flag = true;
    } else if (arg == "--solve-and-write" && i + 1 < argc) {
      out_solved_dump_path = argv[++i];
    } else if (arg == "--timeout-ms" && i + 1 < argc) {
      solve_timeout_ms = parse_int_strict(argv[++i], "--timeout-ms");
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage:\n"
          << "  ls_opt --input <dump.txt> [--emit-gr <instance.gr>] [--emit-map <vars.map>]\n"
          << "  ls_opt --input-td <in.td> [--check-nice] [--to-nice <out.td>]\n"
          << "  ls_opt --input <dump.txt> --input-td <nice.td> --solve-objective\n"
          << "  ls_opt --input <dump.txt> --input-td <nice.td> --solve-and-write <out.txt>\n"
          << "  Optional (solve modes): --timeout-ms <ms> (default: 600000)\n";
      return 0;
    } else {
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }
  }

  const bool solve_write_flag = !out_solved_dump_path.empty();
  if (solve_objective_flag || solve_write_flag) {
    if (input_path.empty() || input_td_path.empty()) {
      throw std::runtime_error("solve modes require both --input and --input-td");
    }
    if (!out_gr_path.empty() || !out_map_path.empty() || !out_nice_td_path.empty()) {
      throw std::runtime_error("solve modes cannot be combined with emit/convert outputs");
    }

    ParsedInstance inst = parse_dump_file(input_path);
    validate_instance(inst);
    Model model = build_model(inst);

    TD td = parse_td_file(input_td_path);
    validate_td_structure(td);
    const int root_bag = choose_root_bag_id(td);
    RootedTD rooted = root_td(td, root_bag);
    const NiceCheckResult chk = check_nice(rooted);
    if (!chk.ok) {
      throw std::runtime_error("Input TD must be nice for --solve-objective; violation at node " +
                               std::to_string(chk.node) + ": " + chk.reason);
    }

    const SolveResult solve = solve_exact_objective_on_nice_td(model, rooted, solve_write_flag, solve_timeout_ms);
    std::cout << "objective_optimal: " << solve.objective << "\n";
    std::cout << "solve_runtime_ms: " << solve.solve_runtime_ms << "\n";
    std::cout << "dp_peak_table_states: " << solve.dp_peak_table_states << "\n";
    std::cout << "dp_peak_live_states: " << solve.dp_peak_live_states << "\n";
    std::cout << "dp_rss_start_kb: " << solve.dp_rss_start_kb << "\n";
    std::cout << "dp_rss_peak_kb: " << solve.dp_rss_peak_kb << "\n";
    std::cout << "dp_rss_end_kb: " << solve.dp_rss_end_kb << "\n";

    if (solve_write_flag) {
      std::vector<std::pair<std::string, std::string>> layout_sel;
      for (size_t vid = 1; vid < model.vars.size() + 1; ++vid) {
        if (model.vars[vid - 1].type != VarType::X) {
          continue;
        }
        const std::string &op = model.op_name_by_x_var[vid];
        const int32_t val = solve.var_value[vid];
        if (op.empty() || val < 0 ||
            static_cast<size_t>(val) >= model.x_domain_labels[vid].size()) {
          throw std::runtime_error("Recovered solution is incomplete for X variable id " + std::to_string(vid));
        }
        layout_sel.push_back({op, model.x_domain_labels[vid][val]});
      }
      std::sort(layout_sel.begin(), layout_sel.end());
      write_dump_with_layout_selection(input_path, out_solved_dump_path, layout_sel);
      std::cout << "wrote solved dump: " << out_solved_dump_path << "\n";
    }
    return 0;
  }

  const bool dump_mode = !input_path.empty();
  const bool td_mode = !input_td_path.empty();

  if (dump_mode && td_mode) {
    throw std::runtime_error("Use either dump mode (--input) or td mode (--input-td), not both");
  }
  if (!dump_mode && !td_mode) {
    throw std::runtime_error("Missing required arguments: provide --input or --input-td");
  }

  if (dump_mode) {
    ParsedInstance inst = parse_dump_file(input_path);
    validate_instance(inst);

    InteractionGraph G;
    InteractionGraph *g_ptr = nullptr;
    if (!out_gr_path.empty() || !out_map_path.empty()) {
      G = build_interaction_graph(inst);
      g_ptr = &G;
    }

    print_instance_summary(inst, input_path, g_ptr);

    if (!out_gr_path.empty()) {
      write_gr(G, out_gr_path);
      std::cout << "wrote .gr: " << out_gr_path << "\n";
    }
    if (!out_map_path.empty()) {
      write_map(G, out_map_path);
      std::cout << "wrote map: " << out_map_path << "\n";
    }

    return 0;
  }

  TD td = parse_td_file(input_td_path);
  validate_td_structure(td);
  print_td_summary(td, "td-input");

  const int root_bag = choose_root_bag_id(td);
  RootedTD rooted = root_td(td, root_bag);

  if (check_nice_flag) {
    const NiceCheckResult chk = check_nice(rooted);
    if (chk.ok) {
      std::cout << "nice-check: already nice\n";
    } else {
      std::cout << "nice-check: NOT nice at node " << chk.node << ": " << chk.reason << "\n";
    }
  }

  if (!out_nice_td_path.empty()) {
    RootedTD nice = convert_to_nice(rooted);
    const NiceCheckResult chk2 = check_nice(nice);
    if (!chk2.ok) {
      throw std::runtime_error("Internal error: converted TD is not nice at node " + std::to_string(chk2.node) +
                               ": " + chk2.reason);
    }
    write_rooted_td(nice, td.n_graph_vertices, out_nice_td_path, "td-nice");
    std::cout << "wrote nice td: " << out_nice_td_path << "\n";
  }

  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}

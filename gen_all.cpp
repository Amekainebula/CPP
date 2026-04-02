#include <bits/stdc++.h>
#include <filesystem>
#include <fstream>

using namespace std;
using i64 = long long;
namespace fs = std::filesystem;

const i64 NEG = -(1LL << 60);

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

int rnd(int l, int r)
{
    return uniform_int_distribution<int>(l, r)(rng);
}

i64 rndll(i64 l, i64 r)
{
    return uniform_int_distribution<i64>(l, r)(rng);
}

struct TestCase
{
    int n, m, k, b;
    vector<i64> val; // 1-indexed
    vector<int> col; // 1-indexed
    vector<tuple<int, int, i64>> edges;
};

struct DSU
{
    vector<int> fa;
    DSU(int n = 0) { init(n); }

    void init(int n)
    {
        fa.resize(n + 1);
        iota(fa.begin(), fa.end(), 0);
    }

    int find(int x)
    {
        return fa[x] == x ? x : fa[x] = find(fa[x]);
    }

    void merge(int x, int y)
    {
        x = find(x);
        y = find(y);
        if (x != y)
            fa[x] = y;
    }
};

// =========================
// generator config
// =========================

// mode:
// 1 = random tree
// 2 = chain
// 3 = star
// 4 = multi-component forest
// 5 = random forest
// 6 = empty graph
//
// weightMode:
// 1 = node/edge random in [-1e9, 0]
// 2 = node = 0, edge random
// 3 = node random, edge = 0
// 4 = node = -1e9, edge = -1e9
// 5 = small negative [-20, 0]
// 6 = near zero [-5, 0]

i64 getNodeWeight(int weightMode)
{
    if (weightMode == 1)
        return rndll(-1000000000LL, 0LL);
    if (weightMode == 2)
        return 0LL;
    if (weightMode == 3)
        return rndll(-1000000000LL, 0LL);
    if (weightMode == 4)
        return -1000000000LL;
    if (weightMode == 5)
        return rndll(-20LL, 0LL);
    return rndll(-5LL, 0LL);
}

i64 getEdgeWeight(int weightMode)
{
    if (weightMode == 1)
        return rndll(-1000000000LL, 0LL);
    if (weightMode == 2)
        return rndll(-1000000000LL, 0LL);
    if (weightMode == 3)
        return 0LL;
    if (weightMode == 4)
        return -1000000000LL;
    if (weightMode == 5)
        return rndll(-20LL, 0LL);
    return rndll(-5LL, 0LL);
}

void assign_color(vector<int> &col, int n, int b, int style)
{
    // style:
    // 1 = around half
    // 2 = sum(c)=b
    // 3 = sum(c)=n-b
    // 4 = random valid
    // 5 = exactly half if possible
    col.assign(n + 1, 0);

    int cnt1 = 0;
    if (style == 1)
    {
        cnt1 = n / 2;
        cnt1 = max(cnt1, b);
        cnt1 = min(cnt1, n - b);
    }
    else if (style == 2)
    {
        cnt1 = b;
    }
    else if (style == 3)
    {
        cnt1 = n - b;
    }
    else if (style == 4)
    {
        cnt1 = rnd(b, n - b);
    }
    else
    {
        cnt1 = n / 2;
        cnt1 = max(cnt1, b);
        cnt1 = min(cnt1, n - b);
    }

    vector<int> p(n);
    iota(p.begin(), p.end(), 1);
    shuffle(p.begin(), p.end(), rng);
    for (int i = 0; i < cnt1; i++)
        col[p[i]] = 1;
}

vector<tuple<int, int, i64>> gen_edges_mode(int mode, int n, int weightMode, int density = 95)
{
    vector<tuple<int, int, i64>> edges;

    if (mode == 1)
    {
        // random tree
        for (int i = 2; i <= n; i++)
        {
            int fa = rnd(1, i - 1);
            edges.push_back({fa, i, getEdgeWeight(weightMode)});
        }
    }
    else if (mode == 2)
    {
        // chain
        for (int i = 2; i <= n; i++)
        {
            edges.push_back({i - 1, i, getEdgeWeight(weightMode)});
        }
    }
    else if (mode == 3)
    {
        // star
        for (int i = 2; i <= n; i++)
        {
            edges.push_back({1, i, getEdgeWeight(weightMode)});
        }
    }
    else if (mode == 4)
    {
        // multi-component forest
        int remain = n;
        int cur = 1;
        while (remain > 0)
        {
            int len;
            if (remain <= 10)
                len = remain;
            else
                len = rnd(3, min(remain, 25));

            int L = cur, R = cur + len - 1;
            for (int i = L + 1; i <= R; i++)
            {
                int fa = rnd(L, i - 1);
                edges.push_back({fa, i, getEdgeWeight(weightMode)});
            }
            cur = R + 1;
            remain -= len;
        }
    }
    else if (mode == 5)
    {
        // random forest, density controls how likely to connect
        // density close to 100 => m close to n-1
        for (int i = 2; i <= n; i++)
        {
            if (rnd(1, 100) > density)
                continue;
            int fa = rnd(1, i - 1);
            edges.push_back({fa, i, getEdgeWeight(weightMode)});
        }
    }
    else if (mode == 6)
    {
        // empty graph
    }

    shuffle(edges.begin(), edges.end(), rng);
    return edges;
}

TestCase build_case(
    int n,
    int k,
    int b,
    int mode,
    int weightMode,
    int colorStyle,
    int density = 95)
{
    TestCase tc;
    tc.n = n;
    tc.k = k;
    tc.b = b;
    tc.val.assign(n + 1, 0);
    tc.col.assign(n + 1, 0);

    for (int i = 1; i <= n; i++)
    {
        tc.val[i] = getNodeWeight(weightMode);
    }

    assign_color(tc.col, n, b, colorStyle);
    tc.edges = gen_edges_mode(mode, n, weightMode, density);
    tc.m = (int)tc.edges.size();
    return tc;
}

// =========================
// std solution
// =========================

i64 solve_one(const TestCase &tc)
{
    int n = tc.n, k = tc.k, b = tc.b;
    const vector<i64> &val = tc.val;
    const vector<int> &c = tc.col;

    vector<vector<pair<int, i64>>> adj(n + 1);
    DSU dsu(n);

    for (auto [u, v, w] : tc.edges)
    {
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        dsu.merge(u, v);
    }

    set<int> roots;
    for (int i = 1; i <= n; i++)
    {
        int fi = dsu.find(i);
        if (roots.insert(fi).second)
        {
            adj[0].push_back({i, 0});
            adj[i].push_back({0, 0});
        }
    }

    vector<int> sz(n + 1), one(n + 1);
    vector<vector<vector<array<i64, 2>>>> dp(n + 1);

    function<void(int, int)> dfs = [&](int u, int p)
    {
        if (u == 0)
        {
            sz[u] = 0;
            one[u] = 0;
            dp[u] = vector<vector<array<i64, 2>>>(1, vector<array<i64, 2>>(1, {NEG, NEG}));
            dp[u][0][0][0] = 0;
        }
        else
        {
            sz[u] = 1;
            one[u] = c[u];
            dp[u] = vector<vector<array<i64, 2>>>(2, vector<array<i64, 2>>(2, {NEG, NEG}));
            dp[u][0][0][0] = 0;
            dp[u][1][c[u]][1] = val[u];
        }

        for (auto [v, w] : adj[u])
        {
            if (v == p)
                continue;
            dfs(v, u);

            int lu = min(k, sz[u]);
            int lv = min(k, sz[v]);
            int lsz = min(k, sz[u] + sz[v]);
            int old_one = one[u];
            one[u] = min(k, one[u] + one[v]);

            vector<vector<array<i64, 2>>> ndp(
                lsz + 1,
                vector<array<i64, 2>>(min(lsz, one[u]) + 1, {NEG, NEG}));

            for (int i = 0; i <= lu; i++)
            {
                for (int t = 0; t <= min(i, old_one); t++)
                {
                    for (int j = 0; j <= lv && i + j <= k; j++)
                    {
                        for (int b1 = 0; b1 <= min(j, one[v]); b1++)
                        {
                            if (dp[u][i][t][0] > NEG / 2)
                            {
                                i64 bestv = max(dp[v][j][b1][0], dp[v][j][b1][1]);
                                if (bestv > NEG / 2)
                                {
                                    ndp[i + j][t + b1][0] = max(
                                        ndp[i + j][t + b1][0],
                                        dp[u][i][t][0] + bestv);
                                }
                            }

                            if (dp[u][i][t][1] > NEG / 2)
                            {
                                if (dp[v][j][b1][0] > NEG / 2)
                                {
                                    ndp[i + j][t + b1][1] = max(
                                        ndp[i + j][t + b1][1],
                                        dp[u][i][t][1] + dp[v][j][b1][0]);
                                }
                                if (dp[v][j][b1][1] > NEG / 2)
                                {
                                    ndp[i + j][t + b1][1] = max(
                                        ndp[i + j][t + b1][1],
                                        dp[u][i][t][1] + dp[v][j][b1][1] + w);
                                }
                            }
                        }
                    }
                }
            }

            sz[u] = lsz;
            dp[u].swap(ndp);
        }
    };

    dfs(0, -1);

    i64 ans = NEG;
    for (int t = b; t <= min(k - b, one[0]); t++)
    {
        ans = max(ans, dp[0][k][t][0]);
    }
    return ans;
}

// =========================
// output
// =========================

void write_lf(ofstream &ofs, const string &s)
{
    ofs.write(s.c_str(), (streamsize)s.size());
    ofs.put('\n');
}

void write_case(ofstream &ofs, const TestCase &tc)
{
    write_lf(ofs, "1");
    write_lf(ofs, to_string(tc.n) + " " + to_string(tc.m) + " " + to_string(tc.k) + " " + to_string(tc.b));

    {
        string line;
        for (int i = 1; i <= tc.n; i++)
        {
            if (i > 1)
                line += ' ';
            line += to_string(tc.val[i]);
        }
        write_lf(ofs, line);
    }

    {
        string line;
        for (int i = 1; i <= tc.n; i++)
        {
            if (i > 1)
                line += ' ';
            line += to_string(tc.col[i]);
        }
        write_lf(ofs, line);
    }

    for (auto [u, v, w] : tc.edges)
    {
        write_lf(ofs, to_string(u) + " " + to_string(v) + " " + to_string(w));
    }
}

void save_case(int id, const TestCase &tc, const string &tag)
{
    i64 ans = solve_one(tc);

    string dir = "data";
    if (!fs::exists(dir))
        fs::create_directory(dir);

    string in_path = dir + "/" + to_string(id) + ".in";
    string out_path = dir + "/" + to_string(id) + ".out";

    ofstream fin(in_path, ios::binary);
    ofstream fout(out_path, ios::binary);

    write_case(fin, tc);
    write_lf(fout, to_string(ans));

    fin.close();
    fout.close();

    cout << "[+] #" << id
         << " [" << tag << "]"
         << " n=" << tc.n
         << " m=" << tc.m
         << " k=" << tc.k
         << " b=" << tc.b
         << '\n';
}

// =========================
// layered generators
// =========================

void gen_small_group(int &id)
{
    // 1~10
    // 小数据主要用于基础覆盖与手算检查
    vector<TestCase> all;

    all.push_back(build_case(6, 3, 1, 2, 5, 1));       // small chain
    all.push_back(build_case(7, 5, 2, 3, 5, 1));       // small star
    all.push_back(build_case(8, 4, 1, 1, 5, 4));       // small random tree
    all.push_back(build_case(9, 6, 2, 4, 5, 1));       // small multi-forest
    all.push_back(build_case(10, 7, 0, 5, 5, 4, 80));  // b=0
    all.push_back(build_case(11, 11, 5, 1, 5, 1));     // k=n
    all.push_back(build_case(12, 8, 4, 2, 2, 2));      // sum(c)=b
    all.push_back(build_case(12, 8, 4, 3, 3, 3));      // sum(c)=n-b
    all.push_back(build_case(12, 6, 3, 6, 1, 1));      // empty graph
    all.push_back(build_case(12, 10, 5, 5, 4, 1, 95)); // almost max m with all -1e9

    for (auto &tc : all)
        save_case(id++, tc, "small");
}

void gen_medium_group(int &id)
{
    // 11~20
    // 中数据：n 拉大到 60~120，很多 m 接近 n-1
    vector<TestCase> all;

    all.push_back(build_case(60, 30, 10, 2, 1, 1));       // chain
    all.push_back(build_case(70, 50, 20, 3, 1, 1));       // star
    all.push_back(build_case(80, 60, 0, 1, 1, 4));        // random tree, b=0
    all.push_back(build_case(90, 90, 30, 1, 6, 1));       // k=n
    all.push_back(build_case(100, 70, 35, 4, 1, 2));      // multi forest sum(c)=b
    all.push_back(build_case(110, 80, 40, 4, 1, 3));      // multi forest sum(c)=n-b
    all.push_back(build_case(120, 100, 50, 5, 1, 1, 98)); // random forest near n-1
    all.push_back(build_case(120, 90, 45, 5, 2, 4, 99));  // node=0
    all.push_back(build_case(120, 85, 42, 5, 3, 4, 99));  // edge=0
    all.push_back(build_case(120, 100, 50, 6, 4, 1));     // empty graph + extreme

    for (auto &tc : all)
        save_case(id++, tc, "medium");
}

void gen_large_group(int &id)
{
    // 21~40
    // 大数据：尽量 n=200；大部分 m 尽量接近 199
    vector<TestCase> all;

    all.push_back(build_case(200, 200, 100, 2, 1, 1));    // full chain
    all.push_back(build_case(200, 200, 100, 3, 1, 1));    // full star
    all.push_back(build_case(200, 200, 0, 1, 1, 4));      // b=0 tree
    all.push_back(build_case(200, 200, 100, 1, 4, 1));    // all -1e9 tree
    all.push_back(build_case(200, 180, 90, 1, 2, 2));     // node=0, sum(c)=b
    all.push_back(build_case(200, 180, 90, 1, 3, 3));     // edge=0, sum(c)=n-b
    all.push_back(build_case(200, 150, 75, 4, 1, 1));     // multi-component forest
    all.push_back(build_case(200, 199, 99, 5, 1, 1, 99)); // dense random forest
    all.push_back(build_case(200, 190, 95, 5, 1, 4, 98));
    all.push_back(build_case(200, 170, 85, 5, 6, 1, 97)); // near zero

    all.push_back(build_case(200, 120, 60, 5, 1, 4, 99));
    all.push_back(build_case(200, 160, 80, 5, 4, 1, 100)); // exactly a tree
    all.push_back(build_case(200, 140, 70, 5, 2, 1, 100)); // exactly a tree, node=0
    all.push_back(build_case(200, 140, 70, 5, 3, 1, 100)); // exactly a tree, edge=0
    all.push_back(build_case(200, 200, 100, 6, 1, 1));     // empty graph, k=n
    all.push_back(build_case(200, 1, 0, 1, 1, 4));         // k=1
    all.push_back(build_case(200, 199, 99, 2, 4, 1));      // chain + all -1e9
    all.push_back(build_case(200, 199, 99, 3, 4, 1));      // star + all -1e9
    all.push_back(build_case(200, 150, 75, 4, 4, 1));      // multi forest + all -1e9
    all.push_back(build_case(200, 199, 100, 1, 6, 5));     // tree + near zero + half color

    for (auto &tc : all)
        save_case(id++, tc, "large");
}

void gen_random_forest_group(int &id)
{
    // 41~60
    // 20 个随机森林随机数据，尽量把 n,m 拉满
    for (int t = 1; t <= 20; t++)
    {
        int n = 200;

        int kType = rnd(1, 6);
        int k;
        if (kType == 1)
            k = 200;
        else if (kType == 2)
            k = rnd(180, 200);
        else if (kType == 3)
            k = rnd(140, 200);
        else if (kType == 4)
            k = rnd(80, 200);
        else if (kType == 5)
            k = rnd(1, 200);
        else
            k = rnd(100, 200);

        int bType = rnd(1, 5);
        int b;
        if (bType == 1)
            b = 0;
        else if (bType == 2)
            b = k / 2;
        else if (bType == 3)
            b = rnd(0, min(10, k / 2));
        else if (bType == 4)
            b = rnd(max(0, k / 2 - 10), k / 2);
        else
            b = rnd(0, k / 2);

        int weightMode = rnd(1, 6);
        int colorStyle = rnd(1, 5);

        // 让绝大多数随机森林 m 接近 n-1
        int density;
        int p = rnd(1, 100);
        if (p <= 50)
            density = 100; // tree
        else if (p <= 80)
            density = 99; // very dense forest
        else if (p <= 95)
            density = 98;
        else
            density = 95;

        TestCase tc = build_case(n, k, b, 5, weightMode, colorStyle, density);
        save_case(id++, tc, "random_forest");
    }
}

i64 brute_force(const TestCase &tc)
{
    int n = tc.n;
    i64 ans = NEG;

    for (int mask = 0; mask < (1 << n); mask++)
    {
        int cnt = 0, cnt1 = 0;
        i64 sum = 0;

        for (int i = 1; i <= n; i++)
        {
            if ((mask >> (i - 1)) & 1)
            {
                cnt++;
                cnt1 += tc.col[i];
                sum += tc.val[i];
            }
        }

        int cnt0 = cnt - cnt1;
        if (cnt < tc.k)
            continue;
        if (cnt0 < tc.b || cnt1 < tc.b)
            continue;

        for (auto [u, v, w] : tc.edges)
        {
            if (((mask >> (u - 1)) & 1) && ((mask >> (v - 1)) & 1))
            {
                sum += w;
            }
        }

        ans = max(ans, sum);
    }

    return ans;
}

TestCase build_small_case_no_kn(
    int mode,
    int weightMode,
    int colorStyle,
    int density = 85)
{
    int n = rnd(6, 12);
    int k = rnd(1, n - 1); // 关键：永远不等于 n
    int b = rnd(0, k / 2);
    return build_case(n, k, b, mode, weightMode, colorStyle, density);
}

void save_multi_small_case(int id, const vector<TestCase> &cases, const string &tag)
{
    string dir = "data";
    if (!fs::exists(dir))
        fs::create_directory(dir);

    string in_path = dir + "/" + to_string(id) + ".in";
    string out_path = dir + "/" + to_string(id) + ".out";

    ofstream fin(in_path, ios::binary);
    ofstream fout(out_path, ios::binary);

    write_lf(fin, to_string((int)cases.size()));
    for (auto &tc : cases)
    {
        write_lf(fin, to_string(tc.n) + " " + to_string(tc.m) + " " + to_string(tc.k) + " " + to_string(tc.b));

        {
            string line;
            for (int i = 1; i <= tc.n; i++)
            {
                if (i > 1)
                    line += ' ';
                line += to_string(tc.val[i]);
            }
            write_lf(fin, line);
        }

        {
            string line;
            for (int i = 1; i <= tc.n; i++)
            {
                if (i > 1)
                    line += ' ';
                line += to_string(tc.col[i]);
            }
            write_lf(fin, line);
        }

        for (auto [u, v, w] : tc.edges)
        {
            write_lf(fin, to_string(u) + " " + to_string(v) + " " + to_string(w));
        }
    }

    for (auto &tc : cases)
    {
        write_lf(fout, to_string(brute_force(tc)));
    }

    fin.close();
    fout.close();

    cout << "[+] #" << id
         << " [" << tag << "]"
         << " cases=" << cases.size()
         << '\n';
}

void gen_extra_small_20_group(int &id)
{
    // 81~85
    for (int file = 1; file <= 5; file++)
    {
        vector<TestCase> cases;
        cases.reserve(20);

        for (int t = 1; t <= 20; t++)
        {
            TestCase tc;
            int pick = rnd(1, 3);

            if (pick == 1)
            {
                // tree
                int weightMode = rnd(1, 6);
                int colorStyle = rnd(1, 5);
                tc = build_small_case_no_kn(1, weightMode, colorStyle, 100);
            }
            else if (pick == 2)
            {
                // random forest
                int weightMode = rnd(1, 6);
                int colorStyle = rnd(1, 5);
                int densityPick = rnd(1, 4);
                int density;
                if (densityPick == 1)
                    density = 35;
                else if (densityPick == 2)
                    density = 60;
                else if (densityPick == 3)
                    density = 85;
                else
                    density = 100;
                tc = build_small_case_no_kn(5, weightMode, colorStyle, density);
            }
            else
            {
                // empty graph
                int weightMode = rnd(1, 6);
                int colorStyle = rnd(1, 5);
                tc = build_small_case_no_kn(6, weightMode, colorStyle, 0);
            }

            // 双保险
            if (tc.k == tc.n)
                tc.k = max(1, tc.n - 1);
            tc.b = min(tc.b, tc.k / 2);

            cases.push_back(tc);
        }

        save_multi_small_case(id++, cases, "small20_simple");
    }
}

void gen_more_large_random_forest_15(int &id)
{
    // 再加 15 个大范围随机森林
    for (int t = 1; t <= 15; t++)
    {
        int n = 200;

        // k 随机，但不等于 n，且尽量分布开
        int kType = rnd(1, 5);
        int k;
        if (kType == 1)
            k = rnd(120, 160);
        else if (kType == 2)
            k = rnd(140, 180);
        else if (kType == 3)
            k = rnd(80, 150);
        else if (kType == 4)
            k = rnd(50, 120);
        else
            k = rnd(100, 199); // 关键：不取 200

        if (k >= n)
            k = n - 1; // 双保险

        // b 不要太接近 0，也不要太接近 k/2
        int half = k / 2;
        int L = max(1, half / 4);
        int R = max(L, half - max(1, half / 4));

        // 如果区间太窄，再稍微放松
        if (L > R)
        {
            L = max(1, half / 3);
            R = max(L, half - max(1, half / 3));
        }
        if (L > R)
        {
            L = 1;
            R = max(1, half - 1);
        }

        int b = rnd(L, R);

        int weightMode = rnd(1, 6);
        int colorStyle = rnd(1, 5);

        // 随机森林，但大多数 m 比较大
        int densityPick = rnd(1, 5);
        int density;
        if (densityPick == 1)
            density = 75;
        else if (densityPick == 2)
            density = 85;
        else if (densityPick == 3)
            density = 92;
        else if (densityPick == 4)
            density = 97;
        else
            density = 100;

        TestCase tc = build_case(n, k, b, 5, weightMode, colorStyle, density);
        save_case(id++, tc, "large_random_forest_extra");
    }
}

void gen_large_random_m_group(int &id)
{
    // 61~80
    // n 固定 200，m 在 [0,199] 内随机
    for (int t = 1; t <= 20; t++)
    {
        int n = 200;

        int kType = rnd(1, 6);
        int k;
        if (kType == 1)
            k = 200;
        else if (kType == 2)
            k = rnd(180, 200);
        else if (kType == 3)
            k = rnd(120, 200);
        else if (kType == 4)
            k = rnd(50, 200);
        else if (kType == 5)
            k = rnd(1, 200);
        else
            k = rnd(90, 200);

        int bType = rnd(1, 5);
        int b;
        if (bType == 1)
            b = 0;
        else if (bType == 2)
            b = k / 2;
        else if (bType == 3)
            b = rnd(0, min(10, k / 2));
        else if (bType == 4)
            b = rnd(max(0, k / 2 - 10), k / 2);
        else
            b = rnd(0, k / 2);

        int weightMode = rnd(1, 6);
        int colorStyle = rnd(1, 5);

        // 关键：让 m 真随机
        // density 对应每个 i 是否连边，最终 m 大约在 [0,199] 内波动
        int densityMode = rnd(1, 8);
        int density;
        if (densityMode == 1)
            density = 5;
        else if (densityMode == 2)
            density = 15;
        else if (densityMode == 3)
            density = 30;
        else if (densityMode == 4)
            density = 50;
        else if (densityMode == 5)
            density = 70;
        else if (densityMode == 6)
            density = 85;
        else if (densityMode == 7)
            density = 95;
        else
            density = 100;

        TestCase tc = build_case(n, k, b, 5, weightMode, colorStyle, density);
        save_case(id++, tc, "large_random_m");
    }
}

int main()
{
    string dir = "data";
    if (!fs::exists(dir))
        fs::create_directory(dir);

    cout << "[*] Start generating layered test data...\n";

    int id = 1;
    gen_small_group(id);                 // 1~10
    gen_medium_group(id);                // 11~20
    gen_large_group(id);                 // 21~40
    gen_random_forest_group(id);         // 41~60
    gen_large_random_m_group(id);        // 61~80
    gen_extra_small_20_group(id);        // 81~85
    gen_more_large_random_forest_15(id); // 86~100

    cout << "[√] Done. Total files generated: " << (id - 1) << '\n';
    return 0;
}

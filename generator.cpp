#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

int rnd(int l, int r)
{
    return uniform_int_distribution<int>(l, r)(rng);
}

i64 rndll(i64 l, i64 r)
{
    return uniform_int_distribution<i64>(l, r)(rng);
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T = 10;
    cout << T << '\n';

    while (T--)
    {
        // 小数据范围，适合暴力
        int n = 100;
        int k = 20;
        int b = rnd(0, k / 2);

        // 生成颜色，保证两类都至少有 b 个
        vector<int> col(n + 1, 0);
        int cnt1 = rnd(b, n - b);

        vector<int> p(n);
        iota(p.begin(), p.end(), 1);
        shuffle(p.begin(), p.end(), rng);

        for (int i = 0; i < cnt1; i++)
        {
            col[p[i]] = 1;
        }

        // 点权
        vector<i64> val(n + 1);
        for (int i = 1; i <= n; i++)
        {
            val[i] = rndll(-20, 0);
        }

        // 生成森林：
        // 先生成树边候选 (i 连到 [1..i-1] 某点)，再随机取前 m 条
        vector<pair<int, int>> candidate;
        for (int i = 2; i <= n; i++)
        {
            int fa = rnd(1, i - 1);
            candidate.push_back({i, fa});
        }
        shuffle(candidate.begin(), candidate.end(), rng);

        int m = rnd(0, (int)candidate.size());
        candidate.resize(m);

        vector<tuple<int, int, i64>> edges;
        for (auto [u, v] : candidate)
        {
            edges.push_back({u, v, rndll(-20, 0)});
        }

        cout << n << ' ' << m << ' ' << k << ' ' << b << '\n';

        for (int i = 1; i <= n; i++)
        {
            cout << val[i] << (i == n ? '\n' : ' ');
        }

        for (int i = 1; i <= n; i++)
        {
            cout << col[i] << (i == n ? '\n' : ' ');
        }

        for (auto [u, v, w] : edges)
        {
            cout << u << ' ' << v << ' ' << w << '\n';
        }
    }

    return 0;
}

#include <bits/stdc++.h>

using namespace std;

using i64 = long long;

struct adj
{
    i64 m; // 合并后的统一值
    i64 f; // 合并后的总代价
    int c; // 多少个分支
};

adj mergeAdj(adj a, adj b)
{
    if (a.c == 0)
        return b;
    if (b.c == 0)
        return a;
    if (a.m < b.m)
        return {b.m, a.f + b.f + a.c * (b.m - a.m), a.c + b.c};
    else
        return {a.m, a.f + b.f + b.c * (a.m - b.m), a.c + b.c};
}

adj addpoint(i64 m, i64 f, i64 w)
{
    return {m + w, f, 1};
}

void Murasame()
{
    int n;
    i64 ans = -1;
    cin >> n;
    vector<vector<pair<i64, int>>> g(n + 1);
    for (int i = 1; i < n; i++)
    {
        int u, v, w;
        cin >> u >> v >> w;
        g[u].push_back({w, v});
        g[v].push_back({w, u});
    }

    vector<int> fa(n + 1, -1), ord;
    ord.reserve(n);
    stack<int> st;
    st.push(1);
    fa[1] = 0;
    while (!st.empty())
    {
        int u = st.top();
        st.pop();
        ord.push_back(u);
        for (auto &[w, v] : g[u])
        {
            if (fa[v] != -1)
                continue;
            fa[v] = u;
            st.push(v);
        }
    }

    vector<i64> dM(n + 1, 0); // 只看u的子树，从u出发到叶节点的公共长度
    vector<i64> dF(n + 1, 0); // 把u子树内所有路径平衡的最小代价

    for (int i = n - 1; i >= 0; --i)
    {
        int u = ord[i];
        adj tmp = {0, 0, 0};
        for (auto &[w, v] : g[u])
        {
            if (v == fa[u])
                continue;
            tmp = mergeAdj(addpoint(dM[v], dF[v], w), tmp);
        }

        if (tmp.c == 0)
        {
            // 叶子
            dM[u] = dF[u] = 0;
        }
        else
        {
            dM[u] = tmp.m;
            dF[u] = tmp.f;
        }
    }

    vector<adj> ug(n + 1, {0, 0, 0});
    for (int u : ord)
    {
        int N = g[u].size();

        vector<adj> arr(N); // 第 i 个邻居方向，对当前点 u 提供的一个原子分支。
        vector<adj> pre(N + 2);
        vector<adj> suf(N + 2);
        for (int i = 0; i < N; i++)
        {
            auto [w, v] = g[u][i];
            if (v == fa[u])
            {
                arr[i] = ug[u];
            }
            else
            {
                arr[i] = addpoint(dM[v], dF[v], w);
            }
        }

        pre[0] = {0, 0, 0};
        for (int i = 0; i < N; i++)
        {
            pre[i + 1] = mergeAdj(pre[i], arr[i]);
        }

        suf[N] = {0, 0, 0};
        for (int i = N - 1; i >= 0; --i)
        {
            suf[i] = mergeAdj(suf[i + 1], arr[i]);
        }

        i64 ansU = (pre[N].c == 0 ? 0 : pre[N].f);
        if (ans == -1 || ansU < ans)
        {
            ans = ansU;
        }

        for (int i = 0; i < N; ++i)
        {
            auto [w, v] = g[u][i];
            if (v == fa[u])
                continue;

            adj oth = mergeAdj(pre[i], suf[i + 1]);
            i64 m = (oth.c == 0 ? 0 : oth.m);
            i64 f = (oth.c == 0 ? 0 : oth.f);

            ug[v] = addpoint(m, f, w);
        }
    }
    cout << ans << '\n';
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }

    return 0;
}

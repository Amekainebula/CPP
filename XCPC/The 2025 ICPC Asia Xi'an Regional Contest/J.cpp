#include <bits/stdc++.h>
#define int long long

using namespace std;
const int N = 3E5 + 10;
int c[N], dfn[N], out[N], dp[N], sum[N], ms1[N], ms2[N], who1[N];
int cnt;
vector<int> g[N];
void dfs(int u, int fa)
{
    dfn[u] = ++cnt;
    for (int v : g[u])
    {
        if (v == fa)
            continue;
        dfs(v, u);
    }
    out[u] = ++cnt;
}
void cost(int u, int fa)
{
    if (g[u].size() == 1) // 叶子节点
    {
        dp[u] = c[u];
        return;
    }
    for (int v : g[u])
    {
        if (v == fa)
            continue;
        cost(v, u);
    }
    
    int min1 = 2e9, min2 = 2e9;
    for (int v : g[u])
    {
        if (v == fa)
            continue;
        if (dp[v] < min1)
        {
            swap(min1, min2);
            min1 = dp[v];
        }
        else if (dp[v] < min2)
        {
            min2 = dp[v];
        }
    }
    dp[u] = min(c[u], min1 + min2);
}

void pre(int u, int fa)
{
    if (g[u].size() == 1) // 叶子节点
    {
        return;
    }
    vector<pair<int, int>> son;
    for (int v : g[u])
    {
        if (v == fa)
            continue;
        pre(v, u);
        son.push_back({dp[v], v});
    }
    sort(son.begin(), son.end());
    ms1[u] = son[0].first;
    ms2[u] = son[1].first;
    who1[u] = son[0].second;
}

int get(int u, int v)
{
    if (who1[u] != v)
        return ms1[u];
    return ms2[u];
}

void ans(int u, int fa)
{
    for (int v : g[u])
    {
        if (v == fa)
            continue;
        sum[v] = sum[u] + get(u, v);
        ans(v, u);
    }
}
bool is(int u, int v)
{
    return dfn[u] <= dfn[v] && out[v] <= out[u];
}
void solve()
{
    cnt = 0;
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        cin >> c[i];
        dp[i] = dfn[i] = out[i] = sum[i] = 0;
        g[i].clear();
    }
    for (int i = 1; i < n; i++)
    {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    dfs(1, 0);
    cost(1, 0);
    pre(1, 0);
    ans(1, 0);
    for (int i = 1; i <= m; i++)
    {
        int x, y;
        cin >> x >> y;
        if (is(y, x))
        {
            cout << sum[x] - sum[y] << '\n';
        }
        else
        {
            cout << -1 << '\n';
        }
    }
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    int _t = 1;
    cin >> _t;
    while (_t--)
    {
        solve();
    }

    return 0;
}

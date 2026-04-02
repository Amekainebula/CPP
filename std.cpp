#include <bits/stdc++.h>

using namespace std;
using i64 = long long;
const int N = 1005;
const i64 NEG = -(1LL << 60);

int fa[N];
int finds(int x)
{
    return fa[x] == x ? x : fa[x] = finds(fa[x]);
}
void merge(int x, int y)
{
    int fx = finds(x), fy = finds(y);
    if (fx != fy)
    {
        fa[fx] = fy;
    }
}

void solve()
{
    int n, m, k, b;
    cin >> n >> m >> k >> b;
    for (int i = 0; i <= n; i++)
    {
        fa[i] = i;
    }
    vector<i64> val(n + 1);
    vector<int> c(n + 1);
    vector<vector<pair<int, i64>>> adj(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> val[i];
    }
    for (int i = 1; i <= n; i++)
    {
        cin >> c[i];
    }
    for (int i = 1; i <= m; i++)
    {
        int u, v;
        i64 w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        merge(u, v);
    }
    set<int> s;
    for (int i = 1; i <= n; i++)
    {
        if (s.insert(finds(i)).second)
        {
            adj[0].push_back({i, 0});
            adj[i].push_back({0, 0});
        }
    }
    vector<int> sz(n + 1);
    vector<int> one(n + 1);
    vector<vector<vector<array<i64, 2>>>> dp(n + 1);
    // dp[u][j][t][0/1]
    // 在 u 子树中，选 j 个点，其中属性 1 的点有 t 个，u 是否被选时的最大值
    auto dfs = [&](auto &&dfs, int u, int p) -> void
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
        for (auto &[v, w] : adj[u])
        {
            if (v == p)
                continue;
            dfs(dfs, v, u);
            int lu = min(k, sz[u]), lv = min(k, sz[v]), lsz = min(k, sz[u] + sz[v]);
            int lone = one[u];
            one[u] = min(k, one[u] + one[v]);
            vector<vector<array<i64, 2>>> dp2(lsz + 1, vector<array<i64, 2>>(min(lsz, one[u]) + 1, {NEG, NEG}));
            for (int i = 0; i <= lu; i++)
            {
                for (int t = 0; t <= min(i, lone); t++)
                {
                    for (int j = 0; j <= lv && i + j <= k; j++)
                    {
                        for (int b1 = 0; b1 <= min(j, one[v]); b1++)
                        {
                            // u不选
                            if (dp[u][i][t][0] > NEG / 2)
                            {
                                i64 bef = max(dp[v][j][b1][0], dp[v][j][b1][1]);
                                if (bef > NEG / 2)
                                {
                                    dp2[i + j][t + b1][0] = max(dp2[i + j][t + b1][0], dp[u][i][t][0] + bef);
                                }
                            }

                            // u选
                            if (dp[u][i][t][1] > NEG / 2)
                            {
                                // v不选
                                if (dp[v][j][b1][0] > NEG / 2)
                                {
                                    dp2[i + j][t + b1][1] = max(dp2[i + j][t + b1][1], dp[u][i][t][1] + dp[v][j][b1][0]);
                                }

                                // v选
                                if (dp[v][j][b1][1] > NEG / 2)
                                {
                                    dp2[i + j][t + b1][1] = max(dp2[i + j][t + b1][1], dp[u][i][t][1] + dp[v][j][b1][1] + w);
                                }
                            }
                        }
                    }
                }
            }
            sz[u] = lsz;
            dp[u].swap(dp2);
        }
    };
    dfs(dfs, 0, -1);

    i64 ans = NEG;
    for (int t = b; t <= min(k - b, one[0]); t++)
    {
        ans = max(ans, dp[0][k][t][0]);
    }
    cout << ans << '\n';
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    int T;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}
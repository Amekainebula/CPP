#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
using namespace std;
void solve()
{
    int n, m, x;
    cin >> n >> m >> x;
    vector<vector<pii>> g(n * 2 + 1);
    for (int i = 1; i <= m; i++)
    {
        int u, v;
        cin >> u >> v;
        g[u].pb({v, 1});
        g[v + n].pb({u + n, 1});
    }
    for (int i = 1; i <= n; i++)
    {
        g[i].pb({i + n, x});
        g[i + n].pb({i, x});
    }
    auto dijkstra = [&](int s,vector<vector<pii>> g) -> vector<int>
    {
        // 导入邻接表，起始位点
        // 邻接表存储[指向的编号，该边的权值]
        int num = g.size();
        vector<int> d(num + 1, INF);
        vector<bool> vis(num + 1);
        auto cmp = [&](pii p1, pii p2) -> bool
        {
            return p1.second == p2.second ? p1.first < p2.first : p1.second > p2.second;
        };
        priority_queue<pii, vector<pii>, decltype(cmp)> pq(cmp);
        // 堆存储[节点编号，起点到该点的最短距离]
        pq.emplace(s, d[s] = 0);
        while (pq.size())
        {
            auto [now, dis] = pq.top();
            pq.pop();
            if (vis[now])
                continue;
            vis[now] = 1;
            // 扩展
            for (auto [nxt, len] : g[now])
            {
                if (d[now] + len < d[nxt])
                {
                    d[nxt] = d[now] + len;
                    pq.push({nxt, d[nxt]});
                }
            }
        }
        return d;
    };
    auto ans = dijkstra(1, g);
    cout << min(ans[n], ans[n * 2]) << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}
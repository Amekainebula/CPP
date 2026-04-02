#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 1e5 + 5;
vector<pair<int, int>> g[N];
vector<int> dis(N), vis(N);
priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
void dijkstar(int s)
{
    dis[s] = 0;
    pq.push({0, s});
    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (vis[u])
            continue;
        vis[u] = 1;
        for (auto [w, v] : g[u])
        {
            if (dis[v] > dis[u] + w)
            {
                dis[v] = dis[u] + w;
                if (!vis[v])
                    pq.push({dis[v], v});
            }
        }
    }
}
void solve()
{
    int n, m, s;
    cin >> n >> m >> s;
    fill(dis.begin(), dis.end(), INT32_MAX);
    for (int i = 1; i <= m; i++)
    {
        int u, v, w;
        cin >> u >> v >> w;
        g[u].push_back({w, v});
    }
    dijkstar(s);
    for (int i = 1; i <= n; i++)
    {
        cout << dis[i] << " ";
    }
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    solve();
    return 0;
}
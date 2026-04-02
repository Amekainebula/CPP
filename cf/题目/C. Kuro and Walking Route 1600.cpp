#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    int n, x, y;
    cin >> n >> x >> y;
    vector<vector<int>> adj(n + 1);
    vector<bool> vis(n + 1, false);
    ff(i, 1, n - 1)
    {
        int u, v;
        cin >> u >> v;
        adj[u].pb(v);
        adj[v].pb(u);
    }
    map<int, int> mp;
    mp[x] = 0, mp[y] = 0;
    auto dfs = [&](this auto &&dfs, int u, int end) -> void
    {
        if (u == end)
            return;
        vis[u] = true;
        for (auto v : adj[u])
        {
            if (!vis[v])
            {
                mp[(end == x ? y : x)]++;
                dfs(v, end);
            }
        }
    };
    dfs(x, y);
    vis.assign(n + 1, false);
    dfs(y, x);
    cout << n * (n - 1) - (n - mp[x]) * (n - mp[y]) << endl;
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
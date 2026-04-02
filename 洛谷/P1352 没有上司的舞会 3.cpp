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
void dfs(int u, vector<int> &vis, vector<vector<int>> &dp, vector<vector<int>> &g)
{
    vis[u] = 1;
    for (int v : g[u])
    {
        if (vis[v])
            continue;
        dfs(v, vis, dp, g);
        dp[u][1] += dp[v][0];
        dp[u][0] += max(dp[v][0], dp[v][1]);
    }
}
void solve()
{
    int n;
    cin >> n;
    vector<vector<int>> dp(n + 1, vector<int>(2));
    vector<int> du(n + 1), vis(n + 1);
    vector<vector<int>> g(n + 1);
    for (int i = 1; i <= n; i++)
        cin >> dp[i][1];
    for (int i = 1; i < n; i++)
    {
        int u, v;
        cin >> u >> v;
        du[u]++;
        g[v].pb(u);
    }
    for (int i = 1; i <= n; i++)
    {
        if (du[i])
            continue;
        dfs(i, vis, dp, g);
        cout << max(dp[i][0], dp[i][1]) << endl;
        return;
    }
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
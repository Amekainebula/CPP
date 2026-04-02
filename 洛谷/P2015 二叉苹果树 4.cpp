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
struct edge
{
    int to, next, w;
};
int head[1010], tot = 0;
int n, m;
int sz[1010];
edge e[1010];
int dp[1010][1010];
void add(int u, int v, int w)
{
    e[++tot].to = v;
    e[tot].next = head[u];
    e[tot].w = w;
    head[u] = tot;
}
void dfs(int u, int fa)
{
    for (int i = head[u]; i; i = e[i].next)
    {
        int v = e[i].to;
        if (v == fa)
            continue;
        dfs(v, u);
        sz[u] += sz[v] + 1;
        for (int j = min(sz[u], m); j >= 1; j--)
        {
            for (int k = min(sz[v], j - 1); k >= 0; k--)
            {
                dp[u][j] = max(dp[u][j], dp[u][j - k - 1] + dp[v][k] + e[i].w);
            }
        }
    }
}
void solve()
{
    cin >> n >> m;
    ff(i, 1, n - 1)
    {
        int u, v, w;
        cin >> u >> v >> w;
        add(u, v, w);
        add(v, u, w);
    }
    dfs(1, 0);
    cout << dp[1][m] << endl;
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
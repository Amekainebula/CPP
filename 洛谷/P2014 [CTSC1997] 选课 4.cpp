#include <bits/stdc++.h>
#define i64 long long
#define u64 unsigned long long
#define i128 __int128
#define d64 long double
#define ff(x, y, z) for (int(x) = (y); (x) < (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) > (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
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
vc<int> g[100005];
vc<int> t(100005, 0);
int n, m;
int dp[1005][1005];
int dfs(int u)
{
    int p = 1;
    for (auto v : g[u])
    {
        int siz = dfs(v);
        for (int i = min(p, m + 1); i; i--)
            for (int j = 1; j <= siz && i + j <= m + 1; j++)
                dp[u][i + j] = max(dp[u][i + j], dp[u][i] + dp[v][j]); 
        p += siz;
    }
    return p;
}
void solve()
{
    cin >> n >> m;
    ff(i, 1, n + 1)
    {
        int u;
        cin >> u >> dp[i][1];
        g[u].pb(i);
    }
    dfs(0);
    cout << dp[0][m + 1] << endl;
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
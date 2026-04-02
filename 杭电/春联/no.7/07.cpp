#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
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
vc<array<int, 3>> g[100005];
vi dp(100005, 0);
int s, t;
void dfs(int u, int fa)
{
    dp[u] = 0;
    for (auto [v, p, q] : g[u])
    {
        if (v == fa)
            continue;
        dfs(v, u);
        dp[u] += max(0LL, dp[v] + p + q);
    }
}
pair<bool, int> solve(int u, int fa)
{
    if (u == t)
        return make_pair(1, dp[u]);
    int res = 0;
    bool ok = 0;
    for (auto [v, p, q] : g[u])
    {
        if (v == fa)
            continue;
        auto [have, val] = solve(v, u);
        if (have)
            ok = 1, res += p + val;
        else
            res += max(0LL, dp[v] + p + q);
    }
    return make_pair(ok, res);
}
void Murasame()
{
    int n;
    cin >> n;
    ff(i, 1, n) g[i].clear();
    ff(i, 1, n - 1)
    {
        int u, v, p, q;
        cin >> u >> v >> p >> q;
        g[u].pb({v, p, q});
        g[v].pb({u, q, p});
    }
    cin >> s >> t;
    dfs(s, 0);
    auto [have, ans] = solve(s, 0);
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}
#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
int fa[500005];
int finds(int x)
{
    return fa[x] == x ? x : fa[x] = finds(fa[x]);
}
void kruskal()
{
    
}
void solve()
{
    int n, m, k;
    cin >> n >> m >> k;
    for (int i = 1; i <= n; i++)
        fa[i] = i;
    vector<pii> e[n + 1];
    vector<int> vis(n + 1, 0);
    map<pii, int> val;
    for (int i = 1; i <= m; i++)
    {
        int u, v, va;
        cin >> u >> v >> va;
        e[u].pb({v, va});
        e[v].pb({u, va});
    }
    for (int i = 1; i <= k; i++)
    {
        int x;
        cin >> x;
        vis[x] = 1;
    }
    kruskal();
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
        solve();
    }
    return 0;
}
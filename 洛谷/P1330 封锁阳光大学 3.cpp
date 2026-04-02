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
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e4 + 6;
int n, m;
int sum[2], ans;
vvi g(N);
vi vis(N, 0), col(N, 0);
bool dfs(int u, int c)
{
    if (vis[u])
    {
        if (col[u] != c)
            return 0;
        return 1;
    }
    vis[u] = 1;
    col[u] = c;
    sum[c]++;
    bool ok = 1;
    for (auto v : g[u])
    {
        if (!ok)
            break;
        ok = ok && dfs(v, c ^ 1);
    }
    return ok;
}
void Murasame()
{
    cin >> n >> m;
    ff(i, 1, m)
    {
        int u, v;
        cin >> u >> v;
        g[u].pb(v);
        g[v].pb(u);
    }
    ff(i, 1, n)
    {
        if (vis[i])
            continue;
        sum[0] = sum[1] = 0;
        if (!dfs(i, 0))
        {
            cout << "Impossible" << endl;
            return;
        }
        ans += min(sum[0], sum[1]);
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}
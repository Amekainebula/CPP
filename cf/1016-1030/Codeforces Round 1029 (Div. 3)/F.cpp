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
const int mod = 1e9 + 7;
const int N = 2e5 + 6;
vvi g(N);
vi len, per(N);
int lca;
void dfs(int u, int par, int l)
{
    if (g[u].size() > 2)
        lca = l;
    bool ok = 1;
    for (int v : g[u])
    {
        if (v != par)
        {
            dfs(v, u, l + 1);
            ok = 0;
        }
    }
    if (ok)
        len.pb(l);
}
int qpow(int a, int b)
{
    int res = 1;
    while (b)
    {
        if (b & 1)
            res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}
void Murasame()
{
    int n;
    cin >> n;
    ff(i, 1, n) g[i].clear();
    lca = -1;
    len.clear();
    ff(i, 1, n - 1)
    {
        int u, v;
        cin >> u >> v;
        g[u].pb(v);
        g[v].pb(u);
    }
    g[1].pb(0);
    dfs(1, 0, 1);
    if (len.size() > 2)
        cout << 0 << endl;
    else if (len.size() == 1)
        cout << qpow(2, n) << endl;
    else
    {
        int d = abs(len[0] - len[1]);
        int x = d + lca;
        if (d)
            cout << (qpow(2, x) + qpow(2, x - 1)) % mod << endl;
        else
            cout << qpow(2, x + 1) << endl;
    }
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
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
double val[1000005];
int fa[1000005], in[1000005];
int finds(int x)
{
    return x == fa[x] ? x : fa[x] = finds(fa[x]);
}
void merge(int x, int y)
{
    x = finds(x), y = finds(y);
    if (x == y)
        return;
    fa[x] = y;
}
void Murasame()
{
    int n;
    cin >> n;
    ff(i, 1, n) cin >> val[i], fa[i] = i, in[i] = 0;
    vi g[n + 1];
    ff(i, 1, n - 1)
    {
        int u, v;
        cin >> u >> v;
        if (val[finds(u)] >= val[finds(v)] / 2 && val[finds(v)] >= val[finds(u)] / 2)
        {
            merge(u, v);
            continue;
        }
        if (val[finds(u)] >= val[finds(v)] / 2)
        {
            g[finds(u)].pb(finds(v));
            in[finds(u)]++;
        }
        if (val[finds(v)] >= val[finds(u)] / 2)
        {
            g[finds(v)].pb(finds(u));
            in[finds(v)]++;
        }
    }
    int cnt0 = 0;
    bool ok = 1;
    int now = -1;
    ff(i, 1, n)
    {
        if (in[finds(i)] == 0)
        {
            cnt0++;
            now = -1;
        }
        else if (in[finds(i)] != 1)
            ok = 0;
    }
    ff(i, 1, n)
    {
        for (auto j : g[finds(i)])
        {
            cout << finds(i) << " " << j << endl;
        }
    }
    if (ok = 0 || cnt0 != 1)
    {
        // cout << 0 << endl;
    }
    int ans = 0;
    ff(i, 1, n)
    {
        if (finds(i) != now)
            ans++;
        //cout << finds(i) << " " << in[finds(i)] << endl;
    }
    // cout << ans << endl;
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
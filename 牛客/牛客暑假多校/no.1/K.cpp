#include <bits/stdc++.h>
// #define int long long
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
const int N = 2e5 + 6;
vvi g(N + 1), val(N, vi(4, 0));
int n;
vc<pii> p(3 * N);
vi ans(N + 1);
map<pii, int> mp, mp2;
int find(int u, int v)
{
    int res = 0;
    for (auto x : g[v])
    {
        if (x == u)
            return res;
        res++;
    }
    return 0;
}
void dfs(int u, int door)
{
    int next = door % g[u].size() + 1;
    int v = g[u][next - 1];
    int temp = find(u, v) + 1;
    mp[{u, v}]++;
    mp[{v, u}]++;
    if (mp2[{u, next}] || val[u][next - 1])
    {
        for (auto x : mp2)
        {
            if (x.se == 0)
                continue;
            auto [u, cnt] = x.fi;
            val[u][cnt - 1] = mp.size() / 2 + (val[u][next - 1] ? val[v][next - 1] : 0);
        }
        return;
    }
    mp2[{u, next}]++;
    dfs(v, temp);
}
void Murasame()
{
    cin >> n;
    ff(i, 1, n)
    {
        int q;
        cin >> q;
        while (q--)
        {
            int v;
            cin >> v;
            g[i].pb(v);
            // g[v].pb(i);
        }
    }
    ff(i, 1, n)
    {
        if (val[i][0])
            continue;
        mp.clear();
        mp2.clear();
        dfs(i, 0);
    }
    ff(i, 1, n)
    {
        cout << val[i][0] << endl;
    }
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